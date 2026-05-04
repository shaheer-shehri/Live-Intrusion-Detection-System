import pathlib
import platform
from pathlib import Path
from typing import Optional, Tuple

# Fix WindowsPath on Linux — two-pronged approach:
# 1. Redirect the module attribute so string-based pickle lookups get PosixPath.
# 2. Patch __new__ on the ORIGINAL class so direct class-object references in
#    the pickle stream (stored by __reduce__) also work.
if platform.system() != 'Windows':
    _OrigWindowsPath = pathlib.WindowsPath          # save before reassigning
    pathlib.WindowsPath = pathlib.PosixPath         # string-lookup fix

    try:
        _OrigWindowsPath.__new__ = staticmethod(    # direct-ref fix
            lambda cls, *a, **kw: pathlib.PosixPath(*a, **kw)
        )
    except (TypeError, AttributeError):
        pass  # read-only in some builds; string-lookup fix still covers most cases

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
