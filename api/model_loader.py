import pathlib
import platform
from pathlib import Path
from typing import Optional, Tuple

# Redirect WindowsPath → PosixPath for pickles saved on Windows.
# Python 3.13+: pathlib is a package; the pickle stream references
#   pathlib._local.WindowsPath — patch that submodule directly.
# Python < 3.13: pathlib is a single file; patch the module attribute.
if platform.system() != 'Windows':
    if hasattr(pathlib, '_local'):
        pathlib._local.WindowsPath = pathlib._local.PosixPath
    pathlib.WindowsPath = pathlib.PosixPath

import joblib

MODEL_PATH    = Path("models/saved/improved_xgboost_mc.joblib")
PIPELINE_PATH = Path("processed_data_mc/combined_preprocessing_pipeline.joblib")


class _FallbackPipeline:
    def transform(self, data):
        return data


def load_model() -> Tuple[object, Optional[list]]:
    try:
        blob = joblib.load(MODEL_PATH)
    except Exception:
        return None, None
    if isinstance(blob, dict) and "model" in blob:
        return blob["model"], blob.get("class_labels")
    return blob, None


def load_pipeline():
    try:
        return joblib.load(PIPELINE_PATH)
    except Exception:
        return _FallbackPipeline()


model, class_labels = load_model()
pipeline = load_pipeline()

# XGBoost validates column order; cache expected order once at startup.
feature_names: Optional[list] = (
    list(model.feature_names_in_)
    if model is not None and hasattr(model, "feature_names_in_")
    else None
)
