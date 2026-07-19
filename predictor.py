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
MODEL_PATH = MODELS_DIR / "knn_fold_4.joblib"
MODEL_TEST_ACCURACY = 0.994819


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
        self.model = self._load_model(models_dir)

    @staticmethod
    def _load_model(models_dir: Path) -> object:
        bundle = joblib.load(models_dir / MODEL_PATH.name)
        if isinstance(bundle, dict) and "model" in bundle:
            return bundle["model"]
        return bundle

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

    def predict(self, file_obj: BinaryIO) -> PredictionResult:
        feature_vector = self.vectorize_upload(file_obj)
        winning_label = str(self.model.predict(feature_vector)[0])
        confidence_percent = MODEL_TEST_ACCURACY * 100.0

        full_name = self._display_name_from_label(winning_label)
        formulation, brand = self._split_full_name(full_name)

        return PredictionResult(
            full_name=full_name,
            formulation=formulation,
            brand=brand,
            confidence_percent=confidence_percent,
        )
