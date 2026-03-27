import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Literal, Union
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler,PowerTransformer,QuantileTransformer

class NumericalScaler:
    """
    RobustScaler
       - Uses: Median and IQR instead of mean/std
       - Pros: Robust to outliers, handles skewed data
       - Cons: Slightly harder to interpret
       - Decision: RECOMMENDED for this dataset
       - Rationale: UNSW-NB15 has 1-3.5% outliers per column
    """
    def __init__(self,
        strategy: Literal['standard', 'minmax', 'robust', 'power', 'quantile', 'none'] = 'robust',
        skip_columns: Optional[List[str]] = None,
        power_method: str = 'yeo-johnson', quantile_output: str = 'uniform'):
        self.strategy = strategy
        self.skip_columns = set(skip_columns or [])
        self.power_method = power_method
        self.quantile_output = quantile_output
        
        # Fitted scaler and metadata
        self.scaler_ = None
        self.scaled_columns_: List[str] = []
        self.scale_params_: Dict[str, Dict] = {}
        self._is_fitted = False
    
    def fit(self, X: pd.DataFrame) -> 'NumericalScaler':
        if self.strategy == 'none':
            self._is_fitted = True
            return self
        print(f"Fitting NumericalScaler with strategy: {self.strategy}")
        
        # Identify columns to scale
        numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        self.scaled_columns_ = [c for c in numerical_cols if c not in self.skip_columns]
        
        if not self.scaled_columns_:
            print("No numerical columns to scale")
            self._is_fitted = True
            return self
        print(f"  Scaling {len(self.scaled_columns_)} columns")
        print(f"  Skipping {len(self.skip_columns)} columns: {list(self.skip_columns)[:5]}...")
        
        # Initialize scaler based on strategy
        if self.strategy == 'standard':
            self.scaler_ = StandardScaler()
        elif self.strategy == 'minmax':
            self.scaler_ = MinMaxScaler()
        elif self.strategy == 'robust':
            self.scaler_ = RobustScaler()
        elif self.strategy == 'power':
            self.scaler_ = PowerTransformer(method=self.power_method, standardize=True)
        elif self.strategy == 'quantile':
            self.scaler_ = QuantileTransformer(output_distribution=self.quantile_output, random_state=42)
        
        # Fit scaler
        self.scaler_.fit(X[self.scaled_columns_])
        # Store scaling parameters for reporting
        self._store_scale_params()
        self._is_fitted = True
        print("  Scaler fitted successfully")
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self._is_fitted:
            raise RuntimeError("NumericalScaler must be fitted before transform")
        if self.strategy == 'none' or not self.scaled_columns_:
            return X
        X = X.copy()
        
        # Handle columns that exist in X
        cols_to_scale = [c for c in self.scaled_columns_ if c in X.columns]
        if cols_to_scale:
            # For partial column matching, we need to handle carefully
            if len(cols_to_scale) == len(self.scaled_columns_):
                # All columns present - direct transform
                X[cols_to_scale] = self.scaler_.transform(X[cols_to_scale])
            else:
                # Partial columns - need to create dummy data for missing cols
                print(f"Only {len(cols_to_scale)}/{len(self.scaled_columns_)} columns found")
                # Transform only available columns
                scaled_data = self._partial_transform(X, cols_to_scale)
                X[cols_to_scale] = scaled_data
        return X
    
    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(X).transform(X)
    
    def _partial_transform(self, X: pd.DataFrame, cols_to_scale: List[str]) -> np.ndarray:
        """Handle transformation when only partial columns are available."""
        # Get indices of columns that exist
        col_indices = [self.scaled_columns_.index(c) for c in cols_to_scale]
        
        # For StandardScaler and similar
        if hasattr(self.scaler_, 'mean_') and hasattr(self.scaler_, 'scale_'):
            result = (X[cols_to_scale].values - self.scaler_.mean_[col_indices]) / self.scaler_.scale_[col_indices]
        elif hasattr(self.scaler_, 'center_') and hasattr(self.scaler_, 'scale_'):
            # RobustScaler
            result = (X[cols_to_scale].values - self.scaler_.center_[col_indices]) / self.scaler_.scale_[col_indices]
        elif hasattr(self.scaler_, 'data_min_') and hasattr(self.scaler_, 'data_range_'):
            # MinMaxScaler
            result = (X[cols_to_scale].values - self.scaler_.data_min_[col_indices]) / self.scaler_.data_range_[col_indices]
        else:
            # For other scalers, transform full and take relevant columns
            full_data = np.zeros((len(X), len(self.scaled_columns_)))
            for i, col in enumerate(cols_to_scale):
                full_data[:, self.scaled_columns_.index(col)] = X[col].values
            result = self.scaler_.transform(full_data)[:, col_indices]
        return result
    
    def _store_scale_params(self) -> None:
        """Store scaling parameters for reporting."""
        if self.scaler_ is None:
            return
        
        for i, col in enumerate(self.scaled_columns_):
            params = {}
            if hasattr(self.scaler_, 'mean_'):
                params['mean'] = self.scaler_.mean_[i]
            if hasattr(self.scaler_, 'var_'):
                params['var'] = self.scaler_.var_[i]
            if hasattr(self.scaler_, 'scale_'):
                params['scale'] = self.scaler_.scale_[i]
            if hasattr(self.scaler_, 'center_'):
                params['center'] = self.scaler_.center_[i]
            if hasattr(self.scaler_, 'data_min_'):
                params['data_min'] = self.scaler_.data_min_[i]
            if hasattr(self.scaler_, 'data_max_'):
                params['data_max'] = self.scaler_.data_max_[i]
            if hasattr(self.scaler_, 'data_range_'):
                params['data_range'] = self.scaler_.data_range_[i]
            
            self.scale_params_[col] = params
    
    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """ Inverse transform scaled features back to original scale. """
        if not self._is_fitted or self.strategy == 'none':
            return X
        X = X.copy()
        cols_to_unscale = [c for c in self.scaled_columns_ if c in X.columns]
        
        if cols_to_unscale and len(cols_to_unscale) == len(self.scaled_columns_):
            X[cols_to_unscale] = self.scaler_.inverse_transform(X[cols_to_unscale])
        return X
    
    def get_scaling_report(self) -> pd.DataFrame:
        if not self.scale_params_:
            return pd.DataFrame()
        
        reports = []
        for col, params in self.scale_params_.items():
            report = {'Column': col, 'Strategy': self.strategy}
            report.update(params)
            reports.append(report)
        
        return pd.DataFrame(reports)
    
    @staticmethod
    def compare_strategies(X: pd.DataFrame, column: str) -> pd.DataFrame:
        results = []
        original = X[column].values.reshape(-1, 1)
        
        strategies = {
            'Original': None,
            'StandardScaler': StandardScaler(),
            'MinMaxScaler': MinMaxScaler(),
            'RobustScaler': RobustScaler(),
            'PowerTransformer': PowerTransformer(method='yeo-johnson')
        }
        
        for name, scaler in strategies.items():
            if scaler is None:
                data = original.flatten()
            else:
                data = scaler.fit_transform(original).flatten()
            
            results.append({
                'Strategy': name,
                'Mean': np.mean(data),
                'Std': np.std(data),
                'Min': np.min(data),
                'Max': np.max(data),
                'Median': np.median(data),
                'Skewness': pd.Series(data).skew(),
                'Kurtosis': pd.Series(data).kurtosis()
            })
        
        return pd.DataFrame(results).round(4)
