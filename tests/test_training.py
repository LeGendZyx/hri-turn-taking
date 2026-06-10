import pandas as pd

from hri_turn_taking.data.dataset import load_windowed_dataset
from hri_turn_taking.data.synthetic import generate_synthetic_turn_taking_data
from hri_turn_taking.models.inference import predict_trp_from_csv
from hri_turn_taking.models.trainer import train_trp_classifier


def test_synthetic_data_loads_as_windowed_dataset(tmp_path) -> None:
    dataset_path = tmp_path / "synthetic.csv"
    frame_df = generate_synthetic_turn_taking_data(
        n_sessions=2,
        duration_s=20.0,
        frame_rate_hz=10,
        random_seed=7,
    )
    frame_df.to_csv(dataset_path, index=False)

    dataset = load_windowed_dataset(dataset_path, window_size=5)

    assert dataset.x.shape[0] == len(frame_df)
    assert dataset.x.shape[1] == 5 * 12
    assert set(dataset.y).issubset({0, 1})


def test_train_and_predict_baseline_model(tmp_path) -> None:
    dataset_path = tmp_path / "synthetic.csv"
    model_path = tmp_path / "trp_model.joblib"
    predictions_path = tmp_path / "predictions.csv"

    frame_df = generate_synthetic_turn_taking_data(
        n_sessions=4,
        duration_s=30.0,
        frame_rate_hz=10,
        random_seed=11,
    )
    frame_df.to_csv(dataset_path, index=False)

    result = train_trp_classifier(
        dataset_path=dataset_path,
        model_path=model_path,
        model_type="random_forest",
        window_size=5,
        random_seed=11,
    )
    predictions = predict_trp_from_csv(
        dataset_path=dataset_path,
        model_path=model_path,
        output_path=predictions_path,
    )

    assert model_path.exists()
    assert predictions_path.exists()
    assert "f1" in result.metrics
    assert len(predictions) == len(pd.read_csv(dataset_path))
