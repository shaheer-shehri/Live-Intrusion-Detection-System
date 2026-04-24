from pathlib import Path
from typing import Optional, Tuple

import joblib

MODEL_PATH = Path("models/saved/improved_xgboost_mc.joblib")
PIPELINE_PATH = Path("processed_data_mc/combined_preprocessing_pipeline.joblib")


class _FallbackPipeline:
    def transform(self, data):
        return data


def load_model() -> Tuple[object, Optional[list]]:
    try:
        blob = joblib.load(MODEL_PATH)
    except ModuleNotFoundError:
        return None, None
    if isinstance(blob, dict) and "model" in blob:
        return blob["model"], blob.get("class_labels")
    return blob, None


def load_pipeline():
    try:
        return joblib.load(PIPELINE_PATH)
    except ModuleNotFoundError:
        return _FallbackPipeline()


model, class_labels = load_model()
pipeline = load_pipeline()