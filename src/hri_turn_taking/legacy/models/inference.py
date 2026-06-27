from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from hri_turn_taking.legacy.data.dataset import build_window_matrix, load_frame_dataframe


def predict_trp_from_csv(
    *,
    dataset_path: str | Path,
    model_path: str | Path,
    output_path: str | Path,
) -> pd.DataFrame:
    artifact = joblib.load(model_path)
    frame_df = load_frame_dataframe(dataset_path)
    x, _groups, timestamps_s, _feature_names = build_window_matrix(
        frame_df,
        feature_columns=artifact["feature_columns"],
        window_size=int(artifact["window_size"]),
    )
    pipeline = artifact["pipeline"]
    probabilities = pipeline.predict_proba(x)[:, 1]
    predictions = pipeline.predict(x).astype(int)

    sorted_df = frame_df.sort_values(["session_id", "timestamp_s"]).reset_index(drop=True)
    result = sorted_df[["session_id", "timestamp_s"]].copy()
    result["trp_probability"] = probabilities
    result["trp_prediction"] = predictions
    if "trp_label" in sorted_df.columns:
        result["trp_label"] = sorted_df["trp_label"].to_numpy(dtype=int)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    return result
