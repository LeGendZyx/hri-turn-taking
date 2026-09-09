"""Participant-structured causal model for multiparty VAP prediction.

Architecture (CLAUDE.md decision #5: light fusion + prediction head, ~130k
params; a GRU temporal variant is kept as the low-cost comparison):

    wide frame features -> per-participant tokens (shared embedding; global
    acoustic features broadcast to every token) -> n_layers blocks of
    [causal temporal attention w/ ALiBi bias (or GRU) -> cross-participant
    attention -> feed-forward] -> shared per-participant heads
    (future-activity bins + auxiliary current-VAD).

What this buys over a flat concatenation encoder:

- **Slot permutation equivariance** — predictions permute with the
  participants, so the arbitrary slot assignment cannot change behaviour
  (unit-tested), and all participant-processing weights are shared (~4x
  supervision per parameter, which matters for a few-hours real dataset).
  3-party and 4-party sessions are handled by the same weights.
- **Explicit cross-participant reasoning** — turn-taking cues are relational
  (who is gazed at, who competes for the floor); the participant-axis
  attention models them directly instead of hoping a flat MLP disentangles
  slots. Absent slots are masked out of this attention, so padded slots can
  never influence present participants (unit-tested).
- **No positional train/inference mismatch** — ALiBi relative bias replaces
  absolute sinusoids (training windows always started at position 0 while
  full-session inference reached position ~4500). The banded causal
  attention also makes chunked streaming inference *exactly* equal to a
  full-session pass (unit-tested), which is what the robot runtime needs.

The model consumes the flat ``(batch, time, features)`` wide-table layout
``[P1 feats | ... | P4 feats | global feats]`` and reshapes internally, so
real extracted features drop in by changing the two feature-count fields.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class VAPModelConfig:
    n_participant_features: int
    n_global_features: int
    n_participants: int = 4
    n_bins: int = 4
    d_model: int = 64
    n_layers: int = 2
    n_heads: int = 4
    dropout: float = 0.1
    encoder: str = "transformer"  # temporal mixing: "transformer" | "gru"
    attention_window: int = 250  # max causal lookback in frames (10 s at 25 fps)

    @property
    def input_dim(self) -> int:
        return self.n_participants * self.n_participant_features + self.n_global_features

    def to_dict(self) -> dict:
        return asdict(self)


def _alibi_slopes(n_heads: int) -> torch.Tensor:
    """Geometric per-head slopes (Press et al., 2022): heads span short/long memory."""
    return torch.tensor([2.0 ** (-8.0 * (i + 1) / n_heads) for i in range(n_heads)])


class _CausalTemporalAttention(nn.Module):
    """Multi-head self-attention over time with ALiBi bias and a banded causal
    window: frame t attends to frames (t - window, t]. Relative bias means no
    absolute-position mismatch between training windows and long sessions."""

    def __init__(self, d_model: int, n_heads: int, dropout: float, window: int) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.window = window
        self.dropout = dropout
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.register_buffer("slopes", _alibi_slopes(n_heads), persistent=False)
        self._bias_cache: tuple[tuple[int, torch.device], torch.Tensor] | None = None

    def _bias(self, n_frames: int, device: torch.device) -> torch.Tensor:
        key = (n_frames, device)
        if self._bias_cache is not None and self._bias_cache[0] == key:
            return self._bias_cache[1]
        position = torch.arange(n_frames, device=device)
        distance = position.unsqueeze(1) - position.unsqueeze(0)  # query - key
        bias = -self.slopes.to(device).view(-1, 1, 1) * distance.clamp(min=0)
        blocked = (distance < 0) | (distance >= self.window)
        bias = bias.masked_fill(blocked.unsqueeze(0), float("-inf")).unsqueeze(0)
        self._bias_cache = (key, bias)  # (1, heads, T, T)
        return bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, n_frames, d_model = x.shape
        query, key, value = self.qkv(x).chunk(3, dim=-1)
        query = query.view(batch, n_frames, self.n_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch, n_frames, self.n_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch, n_frames, self.n_heads, self.head_dim).transpose(1, 2)
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=self._bias(n_frames, x.device),
            dropout_p=self.dropout if self.training else 0.0,
        )
        attended = attended.transpose(1, 2).reshape(batch, n_frames, d_model)
        return self.out(attended)


class _ParticipantAttention(nn.Module):
    """Self-attention across participant slots within each frame (causal in
    time by construction). Absent slots are masked out as keys — except for
    their own query, which keeps every softmax row finite — so padded slots
    cannot influence present participants."""

    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout = dropout
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, present: torch.Tensor | None) -> torch.Tensor:
        batch, n_frames, n_slots, d_model = x.shape
        flat = x.reshape(batch * n_frames, n_slots, d_model)
        query, key, value = self.qkv(flat).chunk(3, dim=-1)
        query = query.view(-1, n_slots, self.n_heads, self.head_dim).transpose(1, 2)
        key = key.view(-1, n_slots, self.n_heads, self.head_dim).transpose(1, 2)
        value = value.view(-1, n_slots, self.n_heads, self.head_dim).transpose(1, 2)

        attn_mask = None
        if present is not None:
            key_ok = present > 0.5  # (batch, n_slots)
            eye = torch.eye(n_slots, dtype=torch.bool, device=x.device)
            allow = key_ok.unsqueeze(1) | eye.unsqueeze(0)  # (batch, query, key)
            bias = torch.zeros(batch, n_slots, n_slots, dtype=x.dtype, device=x.device)
            bias = bias.masked_fill(~allow, float("-inf"))
            attn_mask = (
                bias.unsqueeze(1)
                .unsqueeze(1)
                .expand(batch, n_frames, 1, n_slots, n_slots)
                .reshape(batch * n_frames, 1, n_slots, n_slots)
            )

        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        attended = attended.transpose(1, 2).reshape(batch * n_frames, n_slots, d_model)
        return self.out(attended).view(batch, n_frames, n_slots, d_model)


class _Block(nn.Module):
    """Pre-norm residual block: temporal mixing -> participant mixing -> FFN."""

    def __init__(self, config: VAPModelConfig) -> None:
        super().__init__()
        d_model = config.d_model
        self.norm_temporal = nn.LayerNorm(d_model)
        if config.encoder == "transformer":
            self.temporal: nn.Module = _CausalTemporalAttention(
                d_model, config.n_heads, config.dropout, config.attention_window
            )
            self.gru = None
        else:
            self.temporal = None
            self.gru = nn.GRU(d_model, d_model, batch_first=True)
        self.norm_participant = nn.LayerNorm(d_model)
        self.participant = _ParticipantAttention(d_model, config.n_heads, config.dropout)
        self.norm_ffn = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(d_model * 4, d_model),
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, present: torch.Tensor | None) -> torch.Tensor:
        batch, n_frames, n_slots, d_model = x.shape
        hidden = self.norm_temporal(x)
        hidden = hidden.permute(0, 2, 1, 3).reshape(batch * n_slots, n_frames, d_model)
        if self.temporal is not None:
            hidden = self.temporal(hidden)
        else:
            hidden, _ = self.gru(hidden)
        hidden = hidden.reshape(batch, n_slots, n_frames, d_model).permute(0, 2, 1, 3)
        x = x + self.dropout(hidden)
        x = x + self.dropout(self.participant(self.norm_participant(x), present))
        x = x + self.dropout(self.ffn(self.norm_ffn(x)))
        return x


class MultipartyVAP(nn.Module):
    """Predicts future voice-activity bins for every participant, causally."""

    def __init__(self, config: VAPModelConfig) -> None:
        super().__init__()
        if config.encoder not in {"transformer", "gru"}:
            raise ValueError("encoder must be 'transformer' or 'gru'")
        self.config = config

        self.embed = nn.Sequential(
            nn.Linear(config.n_participant_features + config.n_global_features, config.d_model),
            nn.LayerNorm(config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        self.blocks = nn.ModuleList(_Block(config) for _ in range(config.n_layers))
        self.final_norm = nn.LayerNorm(config.d_model)
        self.bin_head = nn.Linear(config.d_model, config.n_bins)
        self.vad_head = nn.Linear(config.d_model, 1)

    def forward(
        self, features: torch.Tensor, present: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        """features: (batch, time, input_dim) wide layout; present: (batch, n_participants).

        Returns
        -------
        bin_logits: (batch, time, n_participants, n_bins)
        vad_logits: (batch, time, n_participants) — auxiliary current-VAD task
        """
        config = self.config
        batch, n_frames, n_features = features.shape
        if n_features != config.input_dim:
            raise ValueError(f"expected {config.input_dim} input features, got {n_features}")

        n_slot_features = config.n_participants * config.n_participant_features
        per_participant = features[..., :n_slot_features].view(
            batch, n_frames, config.n_participants, config.n_participant_features
        )
        global_block = (
            features[..., n_slot_features:]
            .unsqueeze(2)
            .expand(batch, n_frames, config.n_participants, config.n_global_features)
        )
        x = self.embed(torch.cat([per_participant, global_block], dim=-1))
        if present is not None:
            x = x * present.to(x.dtype).view(batch, 1, config.n_participants, 1)

        for block in self.blocks:
            x = block(x, present)
        x = self.final_norm(x)

        return {"bin_logits": self.bin_head(x), "vad_logits": self.vad_head(x).squeeze(-1)}

    @torch.no_grad()
    def predict_probabilities(
        self, features: torch.Tensor, present: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Convenience wrapper returning sigmoid bin probabilities."""
        self.eval()
        return torch.sigmoid(self(features, present=present)["bin_logits"])
