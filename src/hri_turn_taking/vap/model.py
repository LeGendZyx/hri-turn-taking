"""Causal sequence model for multiparty VAP prediction.

Architecture follows the current VAP literature scaled down to this project's
data budget (CLAUDE.md decision #5: light fusion + prediction head; a GRU
variant is kept as the low-cost comparison architecture):

    frame features -> linear projection -> causal encoder (Transformer | GRU)
                   -> per-participant future-bin logits (+ auxiliary VAD head)

In the deployed system the frame features will be the robot-observable tier
(pretrained ASD probabilities, gaze, head pose, DoA, CPC/prosody embeddings).
The model is feature-agnostic: it only sees a (batch, time, features) tensor.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class VAPModelConfig:
    input_dim: int
    n_participants: int = 4
    n_bins: int = 4
    d_model: int = 64
    n_layers: int = 2
    n_heads: int = 4
    dropout: float = 0.1
    encoder: str = "transformer"  # "transformer" | "gru"
    max_len: int = 8192

    def to_dict(self) -> dict:
        return asdict(self)


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int) -> None:
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        encoding = torch.zeros(max_len, d_model)
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("encoding", encoding, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.encoding[: x.size(1)].unsqueeze(0)


class MultipartyVAP(nn.Module):
    """Predicts future voice-activity bins for every participant, causally."""

    def __init__(self, config: VAPModelConfig) -> None:
        super().__init__()
        if config.encoder not in {"transformer", "gru"}:
            raise ValueError("encoder must be 'transformer' or 'gru'")
        self.config = config

        self.input_projection = nn.Sequential(
            nn.Linear(config.input_dim, config.d_model),
            nn.LayerNorm(config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )

        if config.encoder == "transformer":
            self.positional = SinusoidalPositionalEncoding(config.d_model, config.max_len)
            layer = nn.TransformerEncoderLayer(
                d_model=config.d_model,
                nhead=config.n_heads,
                dim_feedforward=config.d_model * 4,
                dropout=config.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(
                layer, num_layers=config.n_layers, enable_nested_tensor=False
            )
        else:
            self.positional = None
            self.encoder = nn.GRU(
                input_size=config.d_model,
                hidden_size=config.d_model,
                num_layers=config.n_layers,
                batch_first=True,
                dropout=config.dropout if config.n_layers > 1 else 0.0,
            )

        head_out = config.n_participants * config.n_bins
        self.bin_head = nn.Linear(config.d_model, head_out)
        self.vad_head = nn.Linear(config.d_model, config.n_participants)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        """features: (batch, time, input_dim) -> logits dict.

        Returns
        -------
        bin_logits: (batch, time, n_participants, n_bins)
        vad_logits: (batch, time, n_participants) — auxiliary current-VAD task
        """
        batch, time, _ = features.shape
        hidden = self.input_projection(features)

        if self.config.encoder == "transformer":
            hidden = self.positional(hidden)
            causal_mask = nn.Transformer.generate_square_subsequent_mask(
                time, device=features.device
            )
            hidden = self.encoder(hidden, mask=causal_mask, is_causal=True)
        else:
            hidden, _ = self.encoder(hidden)

        bin_logits = self.bin_head(hidden).view(
            batch, time, self.config.n_participants, self.config.n_bins
        )
        vad_logits = self.vad_head(hidden)
        return {"bin_logits": bin_logits, "vad_logits": vad_logits}

    @torch.no_grad()
    def predict_probabilities(self, features: torch.Tensor) -> torch.Tensor:
        """Convenience wrapper returning sigmoid bin probabilities."""
        self.eval()
        return torch.sigmoid(self(features)["bin_logits"])
