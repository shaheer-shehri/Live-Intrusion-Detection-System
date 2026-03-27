import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict, Any, List
from datetime import datetime

from .config import PreprocessingConfig
from .data_loader import DataLoader
from .feature_engineer import FeatureEngineer
from .encoders import CategoricalEncoder
from .scalers import NumericalScaler
from .label_adjustment import build_or_load_adjusted_subset

class PreprocessingPipeline:
    
    def __init__(self, config: Optional[PreprocessingConfig] = None):
        self.config = config or PreprocessingConfig()
        
        # Initialize components
        self.data_loader = DataLoader(data_dir=self.config.data_dir)

        # set feature names if catalog exists so raw files align
        self.data_loader.load_feature_catalog()
        
        protected = getattr(self.config, "protected_features", [])
        self.feature_engineer = FeatureEngineer(
            correlation_threshold=self.config.correlation_threshold,
            mi_threshold=self.config.mi_threshold,
            outlier_lower_pct=self.config.outlier_lower_percentile,
            outlier_upper_pct=self.config.outlier_upper_percentile,
            protected_features=protected,
        )
        self.encoder = CategoricalEncoder(
            strategy=self.config.encoding_strategy,
            categorical_columns=self.config.categorical_columns,
            target_encoding_columns=self.config.target_encoding_columns,
            onehot_encoding_columns=self.config.onehot_encoding_columns,
            high_cardinality_threshold=self.config.high_cardinality_threshold,
        )
        self.scaler = NumericalScaler(
            strategy=self.config.scaling_strategy,
            skip_columns=self.config.skip_scaling_columns,
        )
        self.wide_range_cols_: List[str] = []

        # Pipeline state
        self._is_fitted = False
        self.pipeline_stats_: Dict[str, Any] = {}
        self.drop_by_missing_: List[str] = []
        self.drop_constants_: List[str] = []
        self.numeric_impute_: Dict[str, float] = {}
        self.categorical_impute_: Dict[str, str] = {}

    def _ensure_numeric(self, X: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        for col in cols:
            if col in X.columns:
                X[col] = pd.to_numeric(X[col], errors='coerce')
        return X
        
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> 'PreprocessingPipeline':
        """
        Fit all preprocessing components on training data.
        """
        print("fitting preprocessing pipeline")

        # Ensure IP-like identifiers are removed and fall back to inferred categoricals when analyzer returns none
        base_drop = set(self.config.columns_to_drop) | {"srcip", "dstip"}
        self.config.columns_to_drop = sorted(base_drop)
        if not self.config.categorical_columns:
            inferred_cats = X_train.select_dtypes(include=["object", "string"]).columns
            inferred_cats = [c for c in inferred_cats if c != self.config.attack_category_column and c not in self.config.columns_to_drop]
            self.config.categorical_columns = list(inferred_cats)

        self.pipeline_stats_['initial_shape'] = X_train.shape
        self.pipeline_stats_['initial_columns'] = X_train.columns.tolist()

        # Add domain features before dropping source columns
        X_work = self._add_domain_features(X_train)

        # drop columns with heavy missing and constants using train only
        X_clean = self._drop_missing_and_constants(X_work)
        X_clean = self._downcast_numeric(X_clean)
        X_imputed = self._fit_imputers(X_clean)

        # Step 1: Fit Feature Engineer
        print("\nFeature engineering\n")

        self.feature_engineer.fit(X_imputed, y_train,
            columns_to_drop=self.config.columns_to_drop)
        X_temp = self.feature_engineer.transform(X_imputed)
        self.pipeline_stats_["correlated_removed"] = list(self.feature_engineer.correlated_features_to_remove_)
        self.pipeline_stats_["low_mi_removed"] = list(getattr(self.feature_engineer, "low_mi_features_to_remove_", []))

        # Step 2: Fit Encoder
        print("\nCategorical encoding\n")
        encode_cols = [c for c in self.config.categorical_columns 
                      if c in X_temp.columns and c != self.config.attack_category_column]
        if not encode_cols:
            fallback_objects = X_temp.select_dtypes(include=["object", "string"]).columns
            encode_cols = [c for c in fallback_objects if c != self.config.attack_category_column]
        self.encoder.categorical_columns = encode_cols
        self.encoder.high_cardinality_threshold = self.config.high_cardinality_threshold
        self.encoder.fit(X_temp, y_train)
        X_temp = self.encoder.transform(X_temp)
        # If any object columns survive encoding, factorize them to numeric to avoid downstream float conversion errors
        remaining_obj = X_temp.select_dtypes(include=["object", "string"]).columns
        for col in remaining_obj:
            X_temp[col], _ = pd.factorize(X_temp[col])
        # Step 3: Fit Scaler
        print("\nNumerical scaling\n")

        skip_cols = [c for c in self.config.skip_scaling_columns if c in X_temp.columns]
        self.scaler.skip_columns = set(skip_cols)
        # scale only wide-range numeric columns determined on train
        self.wide_range_cols_ = self._wide_range_columns(X_temp)
        if self.wide_range_cols_:
            cols_to_scale = self.wide_range_cols_
        else:
            cols_to_scale = None
        X_temp = self._ensure_numeric(X_temp, self.wide_range_cols_ or X_temp.columns)
        self.scaler.fit(X_temp[cols_to_scale] if cols_to_scale else X_temp)
        self.pipeline_stats_["scaled_columns"] = self.scaler.scaled_columns_
        self.pipeline_stats_["wide_range_columns"] = self.wide_range_cols_

        # Final guard: ensure all features are numeric for model training
        X_temp = self._coerce_all_numeric(X_temp)

        print("\nPipeline fitted\n")
        self._is_fitted = True
        self.pipeline_stats_['fitted_at'] = datetime.now().isoformat()
        return self
    
    def transform(self, X: pd.DataFrame, apply_imbalance_handling: bool = False,
        y: Optional[pd.Series] = None):
        """
        Transform data using fitted pipeline.apply_imbalance_handling: Whether to apply SMOTE (only for training)Target variable (required if apply_imbalance_handling=True)
        Returns:
            Transformed DataFrame (and resampled y if imbalance handling applied)
        """
        if not self._is_fitted:
            raise RuntimeError("Pipeline must be fitted before transform")
        
        X = X.copy(deep=False)
        X = self._add_domain_features(X)
        X = self._apply_drop_and_impute(X)
        X = self.feature_engineer.transform(X)
        X = self.encoder.transform(X)
        # Ensure numeric before scaling to handle stray string placeholders
        cols_to_coerce = self.wide_range_cols_ if self.wide_range_cols_ else list(self.scaler.scaled_columns_)
        if not cols_to_coerce:
            cols_to_coerce = X.select_dtypes(include=[np.number, "object", "string"]).columns.tolist()
        X = self._ensure_numeric(X, cols_to_coerce)
        if self.wide_range_cols_:
            X[self.wide_range_cols_] = self.scaler.transform(X[self.wide_range_cols_])
        else:
            X = self.scaler.transform(X)

        # Final guard to remove any stray string placeholders
        X = self._coerce_all_numeric(X)
        
        # Step 4: Class Imbalance (optional, training only)
        if apply_imbalance_handling and y is not None:
            X, y = self._apply_imbalance_handling(X, y)
            return X, y
        
        return X

    def _coerce_all_numeric(self, X: pd.DataFrame) -> pd.DataFrame:
        # Apply numeric conversion to every column and fill any introduced NaNs
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce')
        return X.fillna(0)

    def _add_domain_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create duration, packet_rate, and byte_ratio before dropping source columns."""
        X = X.copy(deep=False)

        duration = None
        if "ltime" in X.columns and "stime" in X.columns:
            ltime = pd.to_numeric(X["ltime"], errors="coerce")
            stime = pd.to_numeric(X["stime"], errors="coerce")
            duration = ltime - stime
            duration = duration.replace([np.inf, -np.inf], np.nan)
            duration = duration.clip(lower=0)
            duration = duration.replace(0, np.nan)
            X["duration"] = duration

        if duration is not None and "dpkts" in X.columns:
            dpkts = pd.to_numeric(X["dpkts"], errors="coerce")
            denom = duration.replace(0, np.nan)
            X["packet_rate"] = dpkts / denom

        if "dbytes" in X.columns and "sbytes" in X.columns:
            dbytes = pd.to_numeric(X["dbytes"], errors="coerce")
            sbytes = pd.to_numeric(X["sbytes"], errors="coerce")
            denom = dbytes.replace(0, np.nan)
            X["byte_ratio"] = sbytes / denom

        return X
    
    def fit_transform(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series,
        apply_imbalance_handling: bool = True
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Fit and transform training data in one step. return tuple of (transformed X, resampled y)
        """
        self.fit(X_train, y_train)
        
        if apply_imbalance_handling:
            return self.transform(X_train, apply_imbalance_handling=True, y=y_train)
        else:
            return self.transform(X_train), y_train
    
    def _apply_imbalance_handling(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Apply class imbalance handling strategy which resampled (X, y)
        """
        if self.config.imbalance_strategy == 'none':
            return X, y
        print(f"applying imbalance handling: {self.config.imbalance_strategy}")
        print(f"before class distribution: {y.value_counts().to_dict()}")
        
        if self.config.imbalance_strategy == 'smote':
            from imblearn.over_sampling import SMOTE
            sampler = SMOTE(
                sampling_strategy=self.config.smote_sampling_strategy,
                k_neighbors=self.config.smote_k_neighbors,
                random_state=self.config.smote_random_state
            )
        elif self.config.imbalance_strategy == 'adasyn':
            from imblearn.over_sampling import ADASYN
            sampler = ADASYN(
                sampling_strategy='auto',
                random_state=self.config.smote_random_state
            )
        elif self.config.imbalance_strategy == 'undersample':
            from imblearn.under_sampling import RandomUnderSampler
            sampler = RandomUnderSampler(random_state=self.config.smote_random_state)
        else:
            return X, y
        
        X_resampled, y_resampled = sampler.fit_resample(X, y)
        
        # Convert back to DataFrame
        X_resampled = pd.DataFrame(X_resampled, columns=X.columns)
        y_resampled = pd.Series(y_resampled, name=y.name)
        print(f"after class distribution: {y_resampled.value_counts().to_dict()}")
        
        return X_resampled, y_resampled

    def _drop_missing_and_constants(self, X: pd.DataFrame) -> pd.DataFrame:
        missing_pct = X.isnull().mean()
        self.drop_by_missing_ = missing_pct[missing_pct > self.config.missing_drop_threshold].index.tolist()
        self.drop_constants_ = [c for c in X.columns if X[c].nunique(dropna=False) <= 1]
        drop_cols = list(set(self.config.columns_to_drop + self.drop_by_missing_ + self.drop_constants_))
        X_clean = X.drop(columns=drop_cols, errors="ignore")
        self.pipeline_stats_["dropped_missing"] = self.drop_by_missing_
        self.pipeline_stats_["dropped_constant"] = self.drop_constants_
        return X_clean

    def _fit_imputers(self, X: pd.DataFrame) -> pd.DataFrame:
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        cat_cols = X.select_dtypes(include=["object", "string"]).columns
        self.numeric_impute_ = {c: X[c].median() for c in numeric_cols}
        self.categorical_impute_ = {c: X[c].mode(dropna=True).iloc[0] if not X[c].mode(dropna=True).empty else "" for c in cat_cols}
        return self._apply_impute(X)

    def _apply_impute(self, X: pd.DataFrame) -> pd.DataFrame:
        # shallow copy to avoid large deep copies on full dataset
        X = X.copy(deep=False)
        for c, val in self.numeric_impute_.items():
            if c in X.columns:
                X[c] = X[c].fillna(val)
        for c, val in self.categorical_impute_.items():
            if c in X.columns:
                X[c] = X[c].fillna(val)
        return X

    def _apply_drop_and_impute(self, X: pd.DataFrame) -> pd.DataFrame:
        drop_cols = list(set(self.config.columns_to_drop + self.drop_by_missing_ + self.drop_constants_))
        X = X.drop(columns=drop_cols, errors="ignore")
        X = self._downcast_numeric(X)
        return self._apply_impute(X)

    def _wide_range_columns(self, X: pd.DataFrame) -> List[str]:
        cols = []
        for col in X.select_dtypes(include=[np.number]).columns:
            rng = X[col].max() - X[col].min()
            if rng > self.config.wide_range_threshold and col not in self.config.skip_scaling_columns:
                cols.append(col)
        return cols

    def _downcast_numeric(self, X: pd.DataFrame) -> pd.DataFrame:
        """Downcast numeric columns to reduce memory footprint on full dataset."""
        num_cols = X.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            col_data = pd.to_numeric(X[col], errors="coerce")
            if np.issubdtype(col_data.dtype, np.integer):
                X[col] = col_data.astype(np.int32)
            else:
                X[col] = col_data.astype(np.float32)
        return X
    
    def run_full_pipeline(self, apply_imbalance: bool = True):

        print("Running full preprocessing pipeline")

        # Load default train/test data
        train_df, test_df = self.data_loader.load_train_test()

        if self.config.use_label_adjusted_subset:
            adjusted_path = self.config.output_dir / self.config.adjusted_subset_filename
            adjusted_df = build_or_load_adjusted_subset(
                data_dir=self.config.data_dir,
                output_path=adjusted_path,
                normal_ratio=self.config.adjusted_normal_ratio,
                normal_offset=self.config.adjusted_normal_offset,
                random_state=self.config.random_state,
                force_rebuild=self.config.adjusted_subset_force_rebuild,
            )
            train_df = adjusted_df
            self.pipeline_stats_["train_data_source"] = str(adjusted_path)
            self.pipeline_stats_["train_data_source_type"] = "label_adjusted_subset"
            print(f"Using label-adjusted subset for training: {adjusted_path}")
        else:
            self.pipeline_stats_["train_data_source"] = "training_split"
            self.pipeline_stats_["train_data_source_type"] = "default"
        
        # Separate features and target
        target_col = self.config.target_column
        attack_col = self.config.attack_category_column
        
        # For training - drop both target and attack_cat if available
        y_train = train_df[target_col].copy()
        X_train = train_df.drop(columns=[target_col, attack_col], errors='ignore')
        
        # For testing
        y_test = test_df[target_col].copy()
        X_test = test_df.drop(columns=[target_col, attack_col], errors='ignore')
        
        X_train_processed, y_train_processed = self.fit_transform(
            X_train, y_train,
            apply_imbalance_handling=apply_imbalance
        )
        
        # Transform test data (no imbalance handling)
        X_test_processed = self.transform(X_test)
        
        # Update stats
        self.pipeline_stats_['train_shape_before'] = X_train.shape
        self.pipeline_stats_['train_shape_after'] = X_train_processed.shape
        self.pipeline_stats_['test_shape_before'] = X_test.shape
        self.pipeline_stats_['test_shape_after'] = X_test_processed.shape
        self.pipeline_stats_['class_distribution_before'] = y_train.value_counts().to_dict()
        self.pipeline_stats_['class_distribution_after'] = y_train_processed.value_counts().to_dict()
        
        return X_train_processed, y_train_processed, X_test_processed, y_test
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names after preprocessing."""
        if not self._is_fitted:
            raise RuntimeError("Pipeline must be fitted first")
        return list(self.scaler.scaled_columns_) + list(self.encoder.onehot_feature_names_)
    
    def generate_report(self) -> str:
        """
        preprocessing report
        """
        report = []
        report.append(f"generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"correlation threshold: {self.config.correlation_threshold}")
        report.append(f"encoding: {self.config.encoding_strategy}")
        report.append(f"scaling: {self.config.scaling_strategy}")
        report.append(f"imbalance: {self.config.imbalance_strategy}")

        if self.pipeline_stats_:
            if "train_shape_before" in self.pipeline_stats_:
                report.append(f"train shape: {self.pipeline_stats_.get('train_shape_before')} -> {self.pipeline_stats_.get('train_shape_after')}")
            if "test_shape_before" in self.pipeline_stats_:
                report.append(f"test shape: {self.pipeline_stats_.get('test_shape_before')} -> {self.pipeline_stats_.get('test_shape_after')}")
            if "class_distribution_before" in self.pipeline_stats_:
                report.append(f"class balance before: {self.pipeline_stats_.get('class_distribution_before')}")
            if "class_distribution_after" in self.pipeline_stats_:
                report.append(f"class balance after: {self.pipeline_stats_.get('class_distribution_after')}")
            if self.pipeline_stats_.get("dropped_missing"):
                report.append(f"dropped missing>20%: {self.pipeline_stats_.get('dropped_missing')}")
            if self.pipeline_stats_.get("dropped_constant"):
                report.append(f"dropped constant: {self.pipeline_stats_.get('dropped_constant')}")
            if self.pipeline_stats_.get("correlated_removed"):
                report.append(f"high-corr removed: {self.pipeline_stats_.get('correlated_removed')}")
            if self.pipeline_stats_.get("low_mi_removed"):
                report.append(f"low-mi removed: {self.pipeline_stats_.get('low_mi_removed')}")
            if self.pipeline_stats_.get("wide_range_columns"):
                report.append(f"scaled wide-range cols: {self.pipeline_stats_.get('wide_range_columns')}")
            if self.pipeline_stats_.get("scaled_columns"):
                report.append(f"scaled total cols: {len(self.pipeline_stats_.get('scaled_columns'))}")

        if self._is_fitted:
            enc_report = self.encoder.get_encoding_report()
            if not enc_report.empty:
                report.append("encoding details:")
                report.append(enc_report.to_string(index=False))

        return "\n".join(report)
