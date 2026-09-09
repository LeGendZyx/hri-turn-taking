"""Robot-observable probability baselines for participant-wise VAP.

Both baselines consume only the per-face ASD probability already present in
``SessionTensors.features``.  Gold per-lavalier VAD remains available on the
session solely for labels and event evaluation and is never read here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hri_turn_taking.vap.dataset import (
    TRANSFORMED_GLOBAL_FEATURES,
    TRANSFORMED_PARTICIPANT_FEATURES,
    SessionTensors,
)
from hri_turn_taking.vap.labels import DEFAULT_LABEL_CONFIG, VAPLabelConfig


def _asd_probabilities(session: SessionTensors) -> np.ndarray:
    """Extract raw per-slot ASD inputs without consulting ``session.vad``."""
    try:
        asd_index = TRANSFORMED_PARTICIPANT_FEATURES.index("asd_prob")
    except ValueError as exc:  # defensive guard if the canonical schema changes
        raise ValueError("The canonical participant inputs do not contain asd_prob") from exc

    n_frames = session.features.shape[0]
    n_participants = session.present.shape[0]
    n_participant_features = len(TRANSFORMED_PARTICIPANT_FEATURES)
    expected_width = (
        n_participants * n_participant_features + len(TRANSFORMED_GLOBAL_FEATURES)
    )
    if session.features.ndim != 2 or session.features.shape[1] != expected_width:
        raise ValueError(
            "session.features does not match the canonical transformed participant layout"
        )
    participant_block = session.features[
        :, : n_participants * n_participant_features
    ].reshape(n_frames, n_participants, n_participant_features)
    asd = np.clip(participant_block[:, :, asd_index], 0.0, 1.0)
    return asd * (session.present > 0).astype(np.float32)[None, :]


def asd_persistence_probabilities(
    session: SessionTensors,
    *,
    label_config: VAPLabelConfig = DEFAULT_LABEL_CONFIG,
) -> np.ndarray:
    """Project current ASD probability unchanged into every future bin."""
    asd = _asd_probabilities(session)
    return np.repeat(asd[:, :, None], label_config.n_bins, axis=2).astype(np.float32)


def last_speaker_probabilities(
    session: SessionTensors,
    *,
    label_config: VAPLabelConfig = DEFAULT_LABEL_CONFIG,
    asd_threshold: float = 0.5,
) -> np.ndarray:
    """Causally persist the most recent speaker inferred from ASD.

    A unique highest-ASD present slot above ``asd_threshold`` becomes the
    current last speaker.  If several slots tie, an already established last
    speaker is retained when it is among them; otherwise the ambiguous frame
    does not update state.  During silence the one-hot probability is
    persisted into every future bin.  Frames before any unambiguous ASD
    detection remain all-zero.
    """
    if not 0.0 <= asd_threshold <= 1.0:
        raise ValueError("asd_threshold must be within [0, 1]")
    asd = _asd_probabilities(session)
    n_frames, n_participants = asd.shape
    output = np.zeros((n_frames, n_participants, label_config.n_bins), dtype=np.float32)
    last_speaker: int | None = None
    for frame_idx in range(n_frames):
        frame_asd = asd[frame_idx]
        maximum = float(np.max(frame_asd))
        if maximum >= asd_threshold:
            candidates = np.flatnonzero(
                np.isclose(frame_asd, maximum, rtol=1e-7, atol=1e-12)
            )
            if candidates.size == 1:
                last_speaker = int(candidates[0])
            elif last_speaker is not None and last_speaker in candidates:
                # Preserve established causal state instead of resolving the
                # ambiguity in favour of the lowest participant index.
                pass
        if last_speaker is not None:
            output[frame_idx, last_speaker, :] = 1.0
    return output


@dataclass(frozen=True)
class ASDPersistenceBaseline:
    """``evaluate_model``-compatible current-ASD persistence baseline."""

    def predict_session_probabilities(
        self,
        session: SessionTensors,
        *,
        label_config: VAPLabelConfig = DEFAULT_LABEL_CONFIG,
    ) -> np.ndarray:
        return asd_persistence_probabilities(session, label_config=label_config)


@dataclass(frozen=True)
class LastSpeakerBaseline:
    """``evaluate_model``-compatible ASD-derived last-speaker baseline."""

    asd_threshold: float = 0.5

    def predict_session_probabilities(
        self,
        session: SessionTensors,
        *,
        label_config: VAPLabelConfig = DEFAULT_LABEL_CONFIG,
    ) -> np.ndarray:
        return last_speaker_probabilities(
            session,
            label_config=label_config,
            asd_threshold=self.asd_threshold,
        )
