import sys
import pytest
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing.config import PreprocessingConfig
from preprocessing.data_loader import DataLoader
from preprocessing.feature_engineer import FeatureEngineer
from preprocessing.encoders import CategoricalEncoder, TargetEncoder
from preprocessing.scalers import NumericalScaler
from preprocessing.pipeline import PreprocessingPipeline
from preprocessing.training_utils import load_mc_dataset, default_mc_config

DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "processed_data_mc"
MODEL_DIR = PROJECT_ROOT / "models" / "saved"
SPLITS_DIR = OUTPUT_DIR / "data_splits"
RANDOM_STATE = 42


@pytest.fixture(scope="session")
def raw_dataset():
    """Load the full raw dataset once for the entire test session."""
    X, y_str = load_mc_dataset(DATA_DIR, sample_rows=None)
    return X, y_str


@pytest.fixture(scope="session")
def small_dataset():
    """Load a small sample (first 5 000 rows) for fast component tests."""
    loader = DataLoader(DATA_DIR)
    loader.load_feature_catalog()
    df = loader.load_combined_data(nrows=5000)
    df.columns = df.columns.str.strip().str.lower()
    df["label"] = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int)
    df["attack_cat"] = df.get("attack_cat", pd.Series(["Normal"] * len(df))).fillna("Normal").str.strip()

    y = df["attack_cat"]
    X = df.drop(columns=["attack_cat", "label"], errors="ignore")
    y_numeric, _ = pd.factorize(y)
    y_numeric = pd.Series(y_numeric, name="target_mc", index=X.index)
    return X, y_numeric


@pytest.fixture(scope="session")
def train_test_raw():
    """Load the saved raw train / test CSV splits."""
    X_train = pd.read_csv(SPLITS_DIR / "X_train_raw.csv", low_memory=False)
    X_test = pd.read_csv(SPLITS_DIR / "X_test_raw.csv", low_memory=False)

    y_train = X_train.pop("target")
    y_test = X_test.pop("target")
    return X_train, X_test, y_train, y_test


@pytest.fixture(scope="session")
def fitted_pipeline(small_dataset):
    """Fit a PreprocessingPipeline on the small sample."""
    X, y = small_dataset
    config = default_mc_config(DATA_DIR, OUTPUT_DIR)
    pipe = PreprocessingPipeline(config)
    pipe.fit(X, y)
    return pipe


@pytest.fixture(scope="session")
def preprocessed_splits(fitted_pipeline, small_dataset):
    """Transform the small sample through the fitted pipeline."""
    X, y = small_dataset
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE,
    )
    X_train_prep = fitted_pipeline.transform(X_train)
    X_test_prep = fitted_pipeline.transform(X_test)
    return X_train_prep, X_test_prep, y_train, y_test


class TestDataLoading:
    """Verify that raw data loads correctly."""

    def test_data_directory_exists(self):
        assert DATA_DIR.exists(), f"Data directory missing: {DATA_DIR}"

    def test_csv_files_present(self):
        expected = [f"UNSW-NB15_{i}.csv" for i in range(1, 5)]
        for fname in expected:
            assert (DATA_DIR / fname).exists(), f"Missing data file: {fname}"

    def test_feature_catalog_exists(self):
        assert (DATA_DIR / "NUSW-NB15_features.csv").exists()

    def test_loader_loads_data(self):
        loader = DataLoader(DATA_DIR)
        loader.load_feature_catalog()
        df = loader.load_combined_data(nrows=100)
        assert len(df) > 0, "Loaded DataFrame is empty"
        assert df.shape[1] > 10, "Too few columns loaded"

    def test_loader_column_cleaning(self):
        loader = DataLoader(DATA_DIR)
        loader.load_feature_catalog()
        df = loader.load_combined_data(nrows=100)
        for col in df.columns:
            assert col == col.strip().lower(), f"Column '{col}' not cleaned"

    def test_load_mc_dataset_returns_X_y(self, raw_dataset):
        X, y = raw_dataset
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert len(X) == len(y)

    def test_load_mc_dataset_no_target_in_X(self, raw_dataset):
        X, _ = raw_dataset
        assert "attack_cat" not in X.columns
        assert "label" not in X.columns

    def test_dataset_nonzero_rows(self, raw_dataset):
        X, _ = raw_dataset
        assert len(X) > 0

    def test_saved_splits_exist(self):
        for name in ["X_train_raw.csv", "X_test_raw.csv"]:
            assert (SPLITS_DIR / name).exists(), f"Missing split file: {name}"

    def test_saved_splits_load(self, train_test_raw):
        X_train, X_test, y_train, y_test = train_test_raw
        assert len(X_train) > 0
        assert len(X_test) > 0
        assert len(y_train) == len(X_train)
        assert len(y_test) == len(X_test)


class TestTargetEncoder:
    """Tests for the custom TargetEncoder."""

    def test_fit_transform_shape(self):
        cats = pd.Series(["a", "b", "a", "c", "b", "a"])
        y = pd.Series([1, 0, 1, 0, 0, 1])
        enc = TargetEncoder()
        result = enc.fit_transform(cats, y)
        assert len(result) == len(cats)

    def test_unseen_category_gets_global_mean(self):
        cats = pd.Series(["a", "b", "a"])
        y = pd.Series([1, 0, 1])
        enc = TargetEncoder().fit(cats, y)
        result = enc.transform(pd.Series(["a", "z"]))
        assert result.iloc[1] == pytest.approx(enc.global_mean_, abs=1e-6)

    def test_fitted_flag(self):
        enc = TargetEncoder()
        assert not enc._is_fitted
        enc.fit(pd.Series(["a"]), pd.Series([1]))
        assert enc._is_fitted


class TestCategoricalEncoder:
    """Tests for the CategoricalEncoder."""

    def test_fit_transform_reduces_object_cols(self, small_dataset):
        X, y = small_dataset
        obj_cols = X.select_dtypes(include=["object", "string"]).columns.tolist()
        if not obj_cols:
            pytest.skip("No object columns in small dataset")
        enc = CategoricalEncoder(strategy="hybrid", categorical_columns=obj_cols)
        enc.fit(X, y)
        X_enc = enc.transform(X)
        remaining = X_enc.select_dtypes(include=["object", "string"]).columns
        assert len(remaining) <= len(obj_cols), "Encoder should reduce object columns"

    def test_no_columns_does_not_crash(self):
        X = pd.DataFrame({"a": [1, 2, 3]})
        enc = CategoricalEncoder(categorical_columns=[])
        enc.fit(X)
        result = enc.transform(X)
        assert result.shape == X.shape


class TestNumericalScaler:
    """Tests for the NumericalScaler."""

    def test_robust_scaling_within_range(self):
        rng = np.random.default_rng(42)
        df = pd.DataFrame({"x": rng.normal(100, 50, 200), "y": rng.normal(0, 1, 200)})
        scaler = NumericalScaler(strategy="robust")
        result = scaler.fit_transform(df)
        assert result["x"].std() < df["x"].std(), "Scaling should reduce spread"

    def test_skip_columns_respected(self):
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [10.0, 20.0, 30.0]})
        scaler = NumericalScaler(strategy="standard", skip_columns=["b"])
        result = scaler.fit_transform(df)
        pd.testing.assert_series_equal(result["b"], df["b"], check_names=False)

    def test_strategy_none(self):
        df = pd.DataFrame({"v": [5.0, 10.0, 15.0]})
        scaler = NumericalScaler(strategy="none")
        result = scaler.fit_transform(df)
        pd.testing.assert_frame_equal(result, df)

    def test_fitted_flag(self):
        scaler = NumericalScaler(strategy="standard")
        assert not scaler._is_fitted
        scaler.fit(pd.DataFrame({"a": [1, 2, 3]}))
        assert scaler._is_fitted


class TestFeatureEngineer:
    """Tests for the FeatureEngineer."""

    def test_fit_sets_state(self, small_dataset):
        X, y = small_dataset
        fe = FeatureEngineer(correlation_threshold=0.95)
        fe.fit(X, y)
        assert fe._is_fitted

    def test_transform_drops_correlated_cols(self, small_dataset):
        X, y = small_dataset
        fe = FeatureEngineer(correlation_threshold=0.80)
        fe.fit(X, y)
        X_out = fe.transform(X)
        assert X_out.shape[1] <= X.shape[1]

    def test_transform_not_fitted_raises(self):
        fe = FeatureEngineer()
        with pytest.raises(RuntimeError):
            fe.transform(pd.DataFrame({"a": [1]}))


class TestPreprocessingPipeline:
    """Tests for the end-to-end PreprocessingPipeline."""

    def test_pipeline_fit_sets_fitted(self, fitted_pipeline):
        assert fitted_pipeline._is_fitted

    def test_transform_returns_dataframe(self, preprocessed_splits):
        X_train_prep, X_test_prep, _, _ = preprocessed_splits
        assert isinstance(X_train_prep, pd.DataFrame)
        assert isinstance(X_test_prep, pd.DataFrame)

    def test_transform_before_fit_raises(self):
        config = default_mc_config(DATA_DIR, OUTPUT_DIR)
        pipe = PreprocessingPipeline(config)
        with pytest.raises(RuntimeError):
            pipe.transform(pd.DataFrame({"a": [1]}))

    def test_no_nan_after_transform(self, preprocessed_splits):
        X_train_prep, X_test_prep, _, _ = preprocessed_splits
        assert X_train_prep.isna().sum().sum() == 0, "NaNs in train after preprocessing"
        assert X_test_prep.isna().sum().sum() == 0, "NaNs in test after preprocessing"

    def test_all_numeric_after_transform(self, preprocessed_splits):
        X_train_prep, X_test_prep, _, _ = preprocessed_splits
        for df, label in [(X_train_prep, "train"), (X_test_prep, "test")]:
            obj_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
            assert len(obj_cols) == 0, f"Non-numeric columns survive in {label}: {obj_cols}"

    def test_pipeline_stats_populated(self, fitted_pipeline):
        stats = fitted_pipeline.pipeline_stats_
        assert "initial_shape" in stats
        assert "initial_columns" in stats

    def test_fit_transform_returns_tuple(self, small_dataset):
        X, y = small_dataset
        config = default_mc_config(DATA_DIR, OUTPUT_DIR)
        pipe = PreprocessingPipeline(config)
        result = pipe.fit_transform(X, y, apply_imbalance_handling=False)
        assert isinstance(result, tuple) and len(result) == 2


class TestPredictionShape:
    """Verify saved models produce correct prediction shapes."""

    @pytest.fixture(scope="class")
    def loaded_models(self):
        models = {}
        for name in ["improved_random_forest_mc.joblib", "improved_xgboost_mc.joblib"]:
            path = MODEL_DIR / name
            if path.exists():
                artifact = joblib.load(path)
                models[name] = artifact
        if not models:
            pytest.skip("No saved model files found")
        return models

    @pytest.fixture(scope="class")
    def test_data_preprocessed(self):
        path = SPLITS_DIR / "X_test_preprocessed.csv"
        if not path.exists():
            pytest.skip("Preprocessed test split not found")
        df = pd.read_csv(path)
        # Drop target column so only features remain for prediction
        if "target" in df.columns:
            df = df.drop(columns=["target"])
        return df

    def test_prediction_length_matches_input(self, loaded_models, test_data_preprocessed):
        X = test_data_preprocessed
        for name, artifact in loaded_models.items():
            model = artifact["model"]
            preds = model.predict(X)
            assert len(preds) == len(X), f"{name}: prediction length mismatch"

    def test_proba_shape(self, loaded_models, test_data_preprocessed):
        X = test_data_preprocessed
        for name, artifact in loaded_models.items():
            model = artifact["model"]
            n_classes = artifact["n_classes"]
            proba = model.predict_proba(X)
            assert proba.shape == (len(X), n_classes), (
                f"{name}: proba shape {proba.shape} expected ({len(X)}, {n_classes})"
            )

    def test_predictions_within_class_range(self, loaded_models, test_data_preprocessed):
        X = test_data_preprocessed
        for name, artifact in loaded_models.items():
            model = artifact["model"]
            n_classes = artifact["n_classes"]
            preds = model.predict(X)
            assert preds.min() >= 0
            assert preds.max() < n_classes, f"{name}: pred {preds.max()} >= n_classes={n_classes}"

    def test_proba_rows_sum_to_one(self, loaded_models, test_data_preprocessed):
        X = test_data_preprocessed
        for name, artifact in loaded_models.items():
            model = artifact["model"]
            proba = model.predict_proba(X)
            row_sums = proba.sum(axis=1)
            np.testing.assert_allclose(row_sums, 1.0, atol=1e-5,
                                       err_msg=f"{name}: proba rows don't sum to 1")


class TestFeatureConsistency:
    """Ensure train and test share the same feature set after preprocessing."""

    def test_same_columns(self, preprocessed_splits):
        X_train_prep, X_test_prep, _, _ = preprocessed_splits
        assert list(X_train_prep.columns) == list(X_test_prep.columns), (
            "Train and test have different columns after preprocessing"
        )

    def test_same_column_count(self, preprocessed_splits):
        X_train_prep, X_test_prep, _, _ = preprocessed_splits
        assert X_train_prep.shape[1] == X_test_prep.shape[1]

    def test_same_dtypes(self, preprocessed_splits):
        X_train_prep, X_test_prep, _, _ = preprocessed_splits
        for col in X_train_prep.columns:
            assert X_train_prep[col].dtype == X_test_prep[col].dtype, (
                f"Dtype mismatch for '{col}': train={X_train_prep[col].dtype}, test={X_test_prep[col].dtype}"
            )

    def test_saved_preprocessed_splits_consistent(self):
        """Check the on-disk preprocessed CSVs share columns."""
        train_path = SPLITS_DIR / "X_train_preprocessed.csv"
        test_path = SPLITS_DIR / "X_test_preprocessed.csv"
        if not train_path.exists() or not test_path.exists():
            pytest.skip("Preprocessed CSV splits not found on disk")
        train_cols = pd.read_csv(train_path, nrows=0).columns.tolist()
        test_cols = pd.read_csv(test_path, nrows=0).columns.tolist()
        assert train_cols == test_cols, "Saved train/test preprocessed CSVs have column mismatch"

    def test_no_target_leakage(self, preprocessed_splits):
        """Target or attack_cat should NOT appear in features."""
        X_train_prep, X_test_prep, _, _ = preprocessed_splits
        leaky = {"target", "target_mc", "label", "attack_cat"}
        for df, label in [(X_train_prep, "train"), (X_test_prep, "test")]:
            found = leaky & set(df.columns)
            assert not found, f"Target leakage in {label}: {found}"


class TestReproducibility:
    """Running the pipeline twice with the same seed must give identical results."""

    def test_pipeline_deterministic(self, small_dataset):
        X, y = small_dataset
        results = []
        for _ in range(2):
            config = default_mc_config(DATA_DIR, OUTPUT_DIR)
            pipe = PreprocessingPipeline(config)
            X_out, y_out = pipe.fit_transform(X.copy(), y.copy(),
                                              apply_imbalance_handling=False)
            results.append((X_out, y_out))

        pd.testing.assert_frame_equal(results[0][0], results[1][0],
                                      check_exact=False, atol=1e-6)
        pd.testing.assert_series_equal(results[0][1], results[1][1])

    def test_transform_idempotent_columns(self, fitted_pipeline, small_dataset):
        """Calling transform twice on the same data should yield the same columns."""
        X, _ = small_dataset
        out1 = fitted_pipeline.transform(X.copy())
        out2 = fitted_pipeline.transform(X.copy())
        assert list(out1.columns) == list(out2.columns)
        pd.testing.assert_frame_equal(out1, out2, check_exact=False, atol=1e-6)

    def test_scaler_reproducibility(self):
        rng = np.random.default_rng(42)
        df = pd.DataFrame({"a": rng.normal(0, 1, 100), "b": rng.normal(5, 10, 100)})
        s1 = NumericalScaler(strategy="robust").fit_transform(df.copy())
        s2 = NumericalScaler(strategy="robust").fit_transform(df.copy())
        pd.testing.assert_frame_equal(s1, s2)

    def test_encoder_reproducibility(self):
        cats = pd.Series(["a", "b", "c", "a", "b", "c"] * 10)
        y = pd.Series([0, 1, 0, 1, 0, 1] * 10)
        e1 = TargetEncoder().fit_transform(cats.copy(), y.copy())
        e2 = TargetEncoder().fit_transform(cats.copy(), y.copy())
        pd.testing.assert_series_equal(e1, e2)

class TestConfig:
    """Sanity-check configuration defaults."""

    def test_default_config_creates(self):
        cfg = PreprocessingConfig()
        assert cfg.correlation_threshold > 0

    def test_mc_config(self):
        cfg = default_mc_config(DATA_DIR, OUTPUT_DIR)
        assert cfg.encoding_strategy == "hybrid"
        assert cfg.scaling_strategy == "robust"
        assert "proto" in cfg.categorical_columns

    def test_invalid_correlation_threshold_raises(self):
        with pytest.raises(AssertionError):
            PreprocessingConfig(correlation_threshold=0)

    def test_invalid_percentile_raises(self):
        with pytest.raises(AssertionError):
            PreprocessingConfig(outlier_lower_percentile=99, outlier_upper_percentile=1)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])