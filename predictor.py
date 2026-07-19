from __future__ import annotations

from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import BinaryIO

import joblib
import numpy as np
import pandas as pd


NUM_METADATA_ROWS = 8
MODELS_DIR = Path(__file__).resolve().parent / "models"
METADATA_PATH = MODELS_DIR / "preprocessing_metadata.joblib"
MODEL_FILES = {
    "knn": "knn_fold_4.joblib",
    "rf": "rf_fold_1.joblib",
    "mlp": "mlp_fold_1.joblib",
    "svm": "svm_fold_2.joblib",
    "xgb": "xgb_fold_4.joblib",
}
MODEL_TEST_ACCURACY = {
    "knn": 0.994819,
    "rf": 0.992754,
    "mlp": 0.991718,
    "svm": 0.983420,
    "xgb": 0.983420,
}


@dataclass
class PredictionResult:
    full_name: str
    formulation: str
    brand: str
    confidence_percent: float


class AyurvedaClassifier:
    def __init__(self, models_dir: Path = MODELS_DIR, metadata_path: Path = METADATA_PATH) -> None:
        metadata = joblib.load(metadata_path)
        self.label_to_full_name = dict(metadata["label_to_full_name"])
        self.universal_wavenumbers = np.asarray(metadata["universal_wavenumbers"], dtype=float)
        self.models = self._load_models(models_dir)

    @staticmethod
    def _load_models(models_dir: Path) -> dict[str, object]:
        models: dict[str, object] = {}
        for model_name, filename in MODEL_FILES.items():
            bundle = joblib.load(models_dir / filename)
            if isinstance(bundle, dict) and "model" in bundle:
                models[model_name] = bundle["model"]
            else:
                models[model_name] = bundle
        return models

    @staticmethod
    def _read_spectroscopy_file(file_obj: BinaryIO) -> pd.DataFrame:
        raw_bytes = file_obj.read()
        text = raw_bytes.decode("utf-8")
        df = pd.read_csv(
            StringIO(text),
            sep="\t",
            skiprows=NUM_METADATA_ROWS,
            header=None,
            names=["Wavenumber", "Intensity"],
            encoding="utf-8",
        )
        df["Wavenumber"] = pd.to_numeric(df["Wavenumber"], errors="coerce")
        df["Intensity"] = pd.to_numeric(df["Intensity"], errors="coerce")
        df = df.dropna(subset=["Wavenumber"]).fillna({"Intensity": 0.0})
        df = df.groupby("Wavenumber", as_index=False)["Intensity"].mean()
        df = df.sort_values("Wavenumber")
        return df

    def vectorize_upload(self, file_obj: BinaryIO) -> np.ndarray:
        df = self._read_spectroscopy_file(file_obj)
        aligned = (
            df.set_index("Wavenumber")
            .reindex(self.universal_wavenumbers, fill_value=0.0)["Intensity"]
            .astype(float)
        )
        return aligned.to_numpy(dtype=float).reshape(1, -1)

    @staticmethod
    def _format_brand_name(raw_brand: str) -> str:
        return raw_brand.replace("_", " / ")

    def _display_name_from_label(self, internal_label: str) -> str:
        if "_" not in internal_label:
            return self.label_to_full_name.get(internal_label, internal_label)
        formulation, raw_brand = internal_label.split("_", 1)
        brand = self._format_brand_name(raw_brand)
        return f"{formulation} ({brand})"

    @staticmethod
    def _split_full_name(full_name: str) -> tuple[str, str]:
        if " (" in full_name and full_name.endswith(")"):
            formulation, brand = full_name[:-1].split(" (", 1)
            return formulation, brand
        return full_name, "Unknown"

    def _predict_votes(self, feature_vector: np.ndarray) -> list[tuple[str, str]]:
        predictions: list[tuple[str, str]] = []
        for model_name, model in self.models.items():
            predictions.append((model_name, str(model.predict(feature_vector)[0])))
        return predictions

    def predict(self, file_obj: BinaryIO) -> PredictionResult:
        feature_vector = self.vectorize_upload(file_obj)
        predictions = self._predict_votes(feature_vector)

        vote_counts: dict[str, int] = {}
        for _, label in predictions:
            vote_counts[label] = vote_counts.get(label, 0) + 1

        sorted_votes = sorted(vote_counts.items(), key=lambda item: (-item[1], item[0]))
        winning_label, _ = sorted_votes[0]
        supporting_models = [
            model_name for model_name, label in predictions if label == winning_label
        ]
        confidence_percent = (
            sum(MODEL_TEST_ACCURACY[model_name] for model_name in supporting_models)
            / len(supporting_models)
            * 100.0
        )

        full_name = self._display_name_from_label(winning_label)
        formulation, brand = self._split_full_name(full_name)

        return PredictionResult(
            full_name=full_name,
            formulation=formulation,
            brand=brand,
            confidence_percent=confidence_percent,
        )
