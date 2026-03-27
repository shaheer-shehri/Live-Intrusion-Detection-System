import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Literal
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


class TargetEncoder:
    """
    Target Encoder for high-cardinality categorical features.
    """
    
    def __init__(self, smoothing: float = 1.0, min_samples: int = 1):
        self.smoothing = smoothing
        self.min_samples = min_samples
        self.encoding_map_: Dict[str, float] = {}
        self.global_mean_: float = 0.0
        self._is_fitted = False
    
    def fit(self, X: pd.Series, y: pd.Series) -> 'TargetEncoder':
        
        self.global_mean_ = y.mean()
        
        # Calculate mean target per category
        df = pd.DataFrame({'cat': X, 'target': y})
        agg = df.groupby('cat')['target'].agg(['mean', 'count'])
        
        # Apply smoothing
        for cat, row in agg.iterrows():
            if row['count'] >= self.min_samples:
                # Smoothed encoding: weighted average of category mean and global mean
                weight = row['count'] / (row['count'] + self.smoothing)
                self.encoding_map_[cat] = weight * row['mean'] + (1 - weight) * self.global_mean_
            else:
                self.encoding_map_[cat] = self.global_mean_
        
        self._is_fitted = True
        return self
    
    def transform(self, X: pd.Series) -> pd.Series:
        """
        Transform categorical feature using fitted encoding.
        """
        if not self._is_fitted:
            raise RuntimeError("TargetEncoder must be fitted before transform")
        
        # Map values, use global mean for unseen categories
        return X.map(self.encoding_map_).fillna(self.global_mean_)
    
    def fit_transform(self, X: pd.Series, y: pd.Series) -> pd.Series:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)


class CategoricalEncoder:
    """
    Unified categorical encoding handler with multiple strategies.
    Conditions:
        Low cardinality (<= 10): One-Hot Encoding
        High cardinality (> 10): Target Encoding
    """
    def __init__(self,
        strategy: Literal['label', 'onehot', 'target', 'hybrid'] = 'hybrid',
        categorical_columns: Optional[List[str]] = None,
        target_encoding_columns: Optional[List[str]] = None,
        onehot_encoding_columns: Optional[List[str]] = None,
        high_cardinality_threshold: int = 10,
        handle_unknown: str = 'ignore'
    ):
        self.strategy = strategy
        self.categorical_columns = categorical_columns
        self.target_encoding_columns = target_encoding_columns or []
        self.onehot_encoding_columns = onehot_encoding_columns or []
        self.high_cardinality_threshold = high_cardinality_threshold
        self.handle_unknown = handle_unknown
        
        # Fitted encoders
        self.label_encoders_: Dict[str, LabelEncoder] = {}
        self.target_encoders_: Dict[str, TargetEncoder] = {}
        self.onehot_encoder_: Optional[OneHotEncoder] = None
        self.onehot_columns_: List[str] = []
        self.onehot_feature_names_: List[str] = []
        
        self._is_fitted = False
        self._fitted_columns: List[str] = []
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'CategoricalEncoder':
        # fit encoder to data and returns self to chain with transform
        print(f"Fitting CategoricalEncoder with strategy: {self.strategy}")
        
        # Auto-detect categorical columns if not specified
        if self.categorical_columns is None:
            self._fitted_columns = X.select_dtypes(include=['object', 'string']).columns.tolist()
        else:
            self._fitted_columns = [c for c in self.categorical_columns if c in X.columns]
        
        if not self._fitted_columns:
            print("No categorical columns found to encode")
            self._is_fitted = True
            return self
        print(f"  Encoding {len(self._fitted_columns)} columns: {self._fitted_columns}")
        
        # Fit based on strategy
        if self.strategy == 'label':
            self._fit_label_encoding(X)
        elif self.strategy == 'onehot':
            self._fit_onehot_encoding(X)
        elif self.strategy == 'target':
            if y is None:
                raise ValueError("Target variable required for target encoding")
            self._fit_target_encoding(X, y)
        elif self.strategy == 'hybrid':
            if y is None:
                raise ValueError("Target variable required for hybrid encoding")
            self._fit_hybrid_encoding(X, y)
        
        self._is_fitted = True
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self._is_fitted:
            raise RuntimeError("CategoricalEncoder must be fitted before transform")
        X = X.copy()
        
        if self.strategy == 'label':
            X = self._transform_label_encoding(X)
        elif self.strategy == 'onehot':
            X = self._transform_onehot_encoding(X)
        elif self.strategy == 'target':
            X = self._transform_target_encoding(X)
        elif self.strategy == 'hybrid':
            X = self._transform_hybrid_encoding(X)
        return X
    
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        # Fit and transform in one step.
        return self.fit(X, y).transform(X)
    
    def _fit_label_encoding(self, X: pd.DataFrame) -> None:
        for col in self._fitted_columns:
            le = LabelEncoder()
            # Fit on all possible values including a placeholder for unknown
            values = X[col].astype(str).unique().tolist()
            le.fit(values)
            self.label_encoders_[col] = le
            
            print(f"\n{col}: LabelEncoder ({len(values)} categories)")
    
    def _fit_onehot_encoding(self, X: pd.DataFrame) -> None:
        
        self.onehot_columns_ = self._fitted_columns
        self.onehot_encoder_ = OneHotEncoder( sparse_output=False,
            handle_unknown='ignore' if self.handle_unknown == 'ignore' else 'error',
            drop='first' ) # Drop first category to avoid multicollinearity
        
        self.onehot_encoder_.fit(X[self.onehot_columns_].astype(str))
        self.onehot_feature_names_ = self.onehot_encoder_.get_feature_names_out(self.onehot_columns_).tolist()
        print(f"\nOneHotEncoder: {len(self.onehot_columns_)} cols -> {len(self.onehot_feature_names_)} features")
    
    def _fit_target_encoding(self, X: pd.DataFrame, y: pd.Series) -> None:
        for col in self._fitted_columns:
            te = TargetEncoder()
            te.fit(X[col].astype(str), y)
            self.target_encoders_[col] = te
            print(f"    {col}: TargetEncoder ({X[col].nunique()} categories)")
    
    def _fit_hybrid_encoding(self, X: pd.DataFrame, y: pd.Series) -> None:
        # Determine columns for each encoding type
        target_cols = []
        onehot_cols = []
        
        for col in self._fitted_columns:
            if col in self.target_encoding_columns:
                target_cols.append(col)
            elif col in self.onehot_encoding_columns:
                onehot_cols.append(col)
            else:
                # Auto-select based on cardinality
                cardinality = X[col].nunique()
                if cardinality > self.high_cardinality_threshold:
                    target_cols.append(col)
                else:
                    onehot_cols.append(col)
        
        # Fit target encoders
        for col in target_cols:
            te = TargetEncoder()
            te.fit(X[col].astype(str), y)
            self.target_encoders_[col] = te
            print(f"\n{col}: TargetEncoder ({X[col].nunique()} categories)")
        
        # Fit one-hot encoders
        if onehot_cols:
            self.onehot_columns_ = onehot_cols
            self.onehot_encoder_ = OneHotEncoder(sparse_output=False,
                handle_unknown='ignore', drop='first' )
            self.onehot_encoder_.fit(X[self.onehot_columns_].astype(str))
            self.onehot_feature_names_ = self.onehot_encoder_.get_feature_names_out(self.onehot_columns_).tolist()
            print(f"\nOneHot columns: {onehot_cols} -> {len(self.onehot_feature_names_)} features")
    
    def _transform_label_encoding(self, X: pd.DataFrame) -> pd.DataFrame:
        for col, le in self.label_encoders_.items():
            if col in X.columns:
                # Handle unseen categories
                X[col] = X[col].astype(str)
                unseen_mask = ~X[col].isin(le.classes_)
                if unseen_mask.any():
                    # Map unseen to -1 (or could use mode)
                    X.loc[unseen_mask, col] = le.classes_[0]  # Map to first class
                X[col] = le.transform(X[col])
        return X
    
    def _transform_onehot_encoding(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.onehot_encoder_ is None:
            return X
        # Transform
        encoded = self.onehot_encoder_.transform(X[self.onehot_columns_].astype(str))
        encoded_df = pd.DataFrame(encoded, columns=self.onehot_feature_names_, index=X.index)
        
        # Drop original columns and add encoded
        X = X.drop(columns=self.onehot_columns_)
        X = pd.concat([X, encoded_df], axis=1)
        
        return X
    
    def _transform_target_encoding(self, X: pd.DataFrame) -> pd.DataFrame:
        for col, te in self.target_encoders_.items():
            if col in X.columns:
                X[col] = te.transform(X[col].astype(str))
        return X
    
    def _transform_hybrid_encoding(self, X: pd.DataFrame) -> pd.DataFrame:
        # Apply target encoding
        X = self._transform_target_encoding(X)
        
        # Apply one-hot encoding
        X = self._transform_onehot_encoding(X)
        
        return X
    
    def get_encoding_report(self) -> pd.DataFrame:
        """Generate encoding report for debugging/documentation."""
        if not self._is_fitted:
            return pd.DataFrame()
        
        report_data = []
        
        for col in self.target_encoders_.keys():
            report_data.append({
                'column': col,
                'encoding_type': 'target',
                'num_categories': len(self.target_encoders_[col].encoding_map_)
            })
        
        for col in self.label_encoders_.keys():
            report_data.append({
                'column': col,
                'encoding_type': 'label',
                'num_categories': len(self.label_encoders_[col].classes_)
            })
        
        if self.onehot_columns_:
            for col in self.onehot_columns_:
                report_data.append({
                    'column': col,
                    'encoding_type': 'onehot',
                    'num_categories': len(self.onehot_feature_names_)
                })
        
        return pd.DataFrame(report_data)