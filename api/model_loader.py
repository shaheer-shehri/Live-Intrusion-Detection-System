import pathlib
import pickle
import platform
from pathlib import Path
from typing import Optional, Tuple

# Fix WindowsPath on Linux — patch pickle.Unpickler.find_class so joblib
# (which inherits from it) redirects WindowsPath → PosixPath at unpickle time.
if platform.system() != 'Windows':
    pathlib.WindowsPath = pathlib.PosixPath

    _orig_find_class = pickle.Unpickler.find_class

    def _patched_find_class(self, module, name):
        if name == 'WindowsPath':
            return pathlib.PosixPath
        return _orig_find_class(self, module, name)

    pickle.Unpickler.find_class = _patched_find_class

import joblib

MODEL_PATH = Path("models/saved/improved_xgboost_mc.joblib")
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
