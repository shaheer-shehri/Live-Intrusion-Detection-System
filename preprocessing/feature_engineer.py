import pandas as pd
import numpy as np
from typing import List, Tuple, Optional, Dict, Set
from sklearn.feature_selection import mutual_info_classif

class FeatureEngineer:
    def __init__( self, correlation_threshold: float = 0.95,
        mi_threshold: float = 0.01, outlier_lower_pct: float = 1.0,
        outlier_upper_pct: float = 99.0,protected_features: Optional[List[str]] = None):

        self.correlation_threshold = correlation_threshold
        self.mi_threshold = mi_threshold
        self.outlier_lower_pct = outlier_lower_pct
        self.outlier_upper_pct = outlier_upper_pct
        self.protected_features = set(protected_features or [])
        
        # Fitted parameters (store for transform)
        self.correlated_features_to_remove_: Set[str] = set()
        self.low_mi_features_to_remove_: Set[str] = set()
        self.outlier_bounds_: Dict[str, Tuple[float, float]] = {}
        self.feature_importance_: Dict[str, float] = {}
        self._is_fitted = False
        
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None,
        columns_to_drop: Optional[List[str]] = None ) -> 'FeatureEngineer':

        print("fitting the feature engineer on training data.")
        # Track initial columns
        initial_columns = set(X.columns)
        
        # 1. Handle pre-specified drops
        if columns_to_drop:
            self.columns_to_drop_ = set(columns_to_drop) & initial_columns
        else:
            self.columns_to_drop_ = set()
        
        # Get working columns
        working_columns = initial_columns - self.columns_to_drop_
        X_work = X[[c for c in X.columns if c in working_columns]]
        
        # 2. Find correlated features to remove
        self.correlated_features_to_remove_ = self._find_correlated_features(X_work)
        
        # 3. Calculate mutual information (if y provided)
        if y is not None:
            self._calculate_mutual_information(X_work, y)
            self.low_mi_features_to_remove_ = self._find_low_mi_features()
        
        # 4. Calculate outlier bounds for numerical columns
        self._calculate_outlier_bounds(X_work)
        self._is_fitted = True
        
        # summary
        total_removed = len(self.columns_to_drop_) + len(self.correlated_features_to_remove_) + len(self.low_mi_features_to_remove_)
        print(f"columns to drop preset: {len(self.columns_to_drop_)}")
        print(f"high correlation removed: {len(self.correlated_features_to_remove_)}")
        print(f"low mi removed: {len(self.low_mi_features_to_remove_)}")
        print(f"total remove count: {total_removed}")
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self._is_fitted:
            raise RuntimeError("FeatureEngineer must be fitted before transform")
        X = X.copy()
        
        # 1. Drop specified columns
        all_to_remove = (self.columns_to_drop_ | self.correlated_features_to_remove_ | self.low_mi_features_to_remove_)
        cols_to_drop = [c for c in all_to_remove if c in X.columns]
        X = X.drop(columns=cols_to_drop, errors='ignore')
        # 2. Cap outliers
        X = self._cap_outliers(X)
        print(f"transformed shape {X.shape}")
        return X
    
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None, columns_to_drop: Optional[List[str]] = None
    ) -> pd.DataFrame:
        return self.fit(X, y, columns_to_drop).transform(X)
    
    def _find_correlated_features(self, X: pd.DataFrame) -> Set[str]:
        # Get numerical columns only
        numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numerical_cols) < 2:
            return set()
        print(f"computing correlation for {len(numerical_cols)} numeric columns")
        # Compute correlation matrix
        corr_matrix = X[numerical_cols].corr().abs()
        
        # Get upper triangle
        upper_tri = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        # Find pairs above threshold
        to_remove = set()
        
        for col in upper_tri.columns:
            for row in upper_tri.index:
                if upper_tri.loc[row, col] > self.correlation_threshold:
                    # Decide which to remove
                    if col in self.protected_features:
                        candidate = row
                    elif row in self.protected_features:
                        candidate = col
                    else:
                        # Remove the one with lower variance
                        if X[col].var() > X[row].var():
                            candidate = row
                        else:
                            candidate = col
                    
                    if candidate not in to_remove:
                        to_remove.add(candidate)
                        print(f"remove {candidate} corr {upper_tri.loc[row, col]:.3f} pair with {row if candidate == col else col}")
        
        return to_remove
    
    def _calculate_mutual_information(self, X: pd.DataFrame, y: pd.Series) -> None:
        # Calculate mutual information between features and target.
        numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        if not numerical_cols:
            return
        
        print(f"computing mutual information for {len(numerical_cols)} numeric columns")
        # Handle missing values for MI calculation
        X_clean = X[numerical_cols].fillna(X[numerical_cols].median())
        mi_scores = mutual_info_classif(X_clean, y, discrete_features=False,random_state=42)
        self.feature_importance_ = dict(zip(numerical_cols, mi_scores))
    
    def _find_low_mi_features(self) -> Set[str]:
        low_mi = set()
        
        for feature, mi in self.feature_importance_.items():
            if mi < self.mi_threshold and feature not in self.protected_features:
                low_mi.add(feature)
                print(f"low mi feature {feature} mi {mi:.4f}")
        
        return low_mi
    
    def _calculate_outlier_bounds(self, X: pd.DataFrame) -> None:
        numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in numerical_cols:
            lower = np.percentile(X[col].dropna(), self.outlier_lower_pct)
            upper = np.percentile(X[col].dropna(), self.outlier_upper_pct)
            self.outlier_bounds_[col] = (lower, upper)
    
    def _cap_outliers(self, X: pd.DataFrame) -> pd.DataFrame:
        # Cap outliers using pre-calculated bounds.
        # Capping (Winsorization) chosen over removal to preserves all samples (important for imbalanced data)
        X = X.copy()
        for col, (lower, upper) in self.outlier_bounds_.items():
            if col in X.columns:
                X[col] = X[col].clip(lower=lower, upper=upper)
        return X
    
    def get_feature_importance_report(self) -> pd.DataFrame:
        # DataFrame with features and their MI (importance) scores, sorted by importance
        if not self.feature_importance_:
            raise RuntimeError("Mutual information not calculated. Call fit() with y parameter.")
        
        df = pd.DataFrame({'Feature': list(self.feature_importance_.keys()),
            'MI_Score': list(self.feature_importance_.values()) })
        
        df = df.sort_values('MI_Score', ascending=False).reset_index(drop=True)
        df['Rank'] = range(1, len(df) + 1)
        df['Will_Remove'] = df['MI_Score'] < self.mi_threshold
        
        return df[['Rank', 'Feature', 'MI_Score', 'Will_Remove']]
    
    def get_correlation_report(self, X: pd.DataFrame) -> pd.DataFrame:
        # DataFrame of highly correlated feature pairs.
        numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        corr_matrix = X[numerical_cols].corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        pairs = []
        for col in upper_tri.columns:
            for row in upper_tri.index:
                if upper_tri.loc[row, col] > self.correlation_threshold:
                    pairs.append({'Feature_1': row, 'Feature_2': col,'Correlation': upper_tri.loc[row, col],
                        'Will_Remove': col if col in self.correlated_features_to_remove_ else row
                    })
        
        return pd.DataFrame(pairs).sort_values('Correlation', ascending=False)
