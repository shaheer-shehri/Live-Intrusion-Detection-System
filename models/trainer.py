import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional
from pathlib import Path
import joblib
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
from xgboost import XGBClassifier
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelTrainer:

    def __init__(self,model_type: str = 'random_forest',
        model_params: Optional[Dict[str, Any]] = None,
        random_state: int = 42,verbose: bool = True):

        self.model_type = model_type
        self.model_params = model_params or {}
        self.random_state = random_state
        self.verbose = verbose
        
        self.model = None
        self.training_stats_: Dict[str, Any] = {}
        self.feature_importance_: Optional[pd.DataFrame] = None
        
    def _create_model(self):
        if self.model_type == 'random_forest':
            default_params = {
                'n_estimators': 100,
                'max_depth': 20,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'max_features': 'sqrt',
                'n_jobs': -1,
                'random_state': self.random_state,
                'class_weight': 'balanced'  # Handle any remaining imbalance
            }
            default_params.update(self.model_params)
            return RandomForestClassifier(**default_params)
        
        elif self.model_type == 'xgboost':
            default_params = {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': self.random_state,
                'n_jobs': -1,
                'scale_pos_weight': 1,
                'eval_metric': 'logloss',
                'verbosity': 0
            }
            default_params.update(self.model_params)
            return XGBClassifier(**default_params)
        
        else:
            raise ValueError(
                f"Unknown model type: {self.model_type}. "
                f"Supported models: 'random_forest', 'xgboost'"
            )
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> 'ModelTrainer':

        if self.verbose:
            print(f"TRAINING {self.model_type.upper()} MODEL")
            print(f"Training samples: {len(X_train):,}")
            print(f"Features: {X_train.shape[1]}")
        
        # Create and train model
        self.model = self._create_model()
        
        start_time = datetime.now()
        self.model.fit(X_train, y_train)
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Store training stats
        self.training_stats_['model_type'] = self.model_type
        self.training_stats_['training_samples'] = len(X_train)
        self.training_stats_['n_features'] = X_train.shape[1]
        self.training_stats_['training_time_seconds'] = training_time
        self.training_stats_['trained_at'] = datetime.now().isoformat()
        
        # Extract feature importance for Random Forest
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance_ = pd.DataFrame({
                'Feature': X_train.columns,
                'Importance': self.model.feature_importances_
            }).sort_values('Importance', ascending=False).reset_index(drop=True)
        
        if self.verbose:
            print(f"Training completed in {training_time:.2f} seconds")
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        if self.model is None:
            raise RuntimeError("Model must be trained before prediction")
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities."""
        if self.model is None:
            raise RuntimeError("Model must be trained before prediction")
        return self.model.predict_proba(X)
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        if self.verbose:
            print("MODEL EVALUATION")
        
        # Predictions
        y_pred = self.predict(X_test)
        proba = self.predict_proba(X_test)

        n_classes = len(np.unique(y_test))
        is_multiclass = n_classes > 2
        avg = 'macro' if is_multiclass else 'binary'

        # Calculate metrics (macro for multiclass)
        if is_multiclass:
            roc_auc = roc_auc_score(y_test, proba, multi_class='ovr', average='macro')
        else:
            roc_auc = roc_auc_score(y_test, proba[:, 1])

        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average=avg, zero_division=0),
            'recall': recall_score(y_test, y_pred, average=avg, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, average=avg, zero_division=0),
            'roc_auc': roc_auc,
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
        
        # Store in training stats
        self.training_stats_['test_metrics'] = {
            k: v for k, v in metrics.items() 
            if k not in ['confusion_matrix', 'classification_report']
        }
        
        if self.verbose:
            self._print_evaluation(metrics, y_test)
        
        return metrics
    
    def _print_evaluation(self, metrics: Dict, y_test: pd.Series) -> None:
        """Print formatted evaluation results."""
        print(f"\nTest Set Size: {len(y_test):,} samples")
        print(f"\n{'Metric':<20} {'Value':>10}")
        print(f"{'Accuracy':<20} {metrics['accuracy']:>10.4f}")
        print(f"{'Precision':<20} {metrics['precision']:>10.4f}")
        print(f"{'Recall':<20} {metrics['recall']:>10.4f}")
        print(f"{'F1-Score':<20} {metrics['f1_score']:>10.4f}")
        print(f"{'ROC-AUC':<20} {metrics['roc_auc']:>10.4f}")
        
        # Per-class metrics
        report = metrics['classification_report']
        n_classes = len([k for k in report.keys() if k.isdigit()])
        
        print("\nPer-Class Performance:")
        print(f"{'Class':<30} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
        
        for class_id in sorted([k for k in report.keys() if k.isdigit()], key=lambda x: int(x)):
            class_metrics = report[class_id]
            print(f"  {class_id:<28} {class_metrics['precision']:>10.4f} "
                  f"{class_metrics['recall']:>10.4f} {class_metrics['f1-score']:>10.4f} "
                  f"{int(class_metrics['support']):>10}")
        
        # Print confusion matrix summary for multi-class (top misclassifications)
        cm = np.array(metrics['confusion_matrix'])
        if n_classes > 2:
            print("\nConfusion Matrix Summary (top misclassifications):")
            # Find top 5 off-diagonal errors
            errors = []
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    if i != j and cm[i, j] > 0:
                        errors.append((i, j, cm[i, j]))
            errors.sort(key=lambda x: x[2], reverse=True)
            for true_class, pred_class, count in errors[:5]:
                print(f"  True class {true_class} → Predicted as {pred_class}: {count:,} samples")
        else:
            # Binary classification - show 2x2 matrix
            print("\nConfusion Matrix:")
            print("                 Predicted")
            print("                 Normal  Attack")
            print(f"Actual Normal    {cm[0,0]:>6}  {cm[0,1]:>6}")
            print(f"Actual Attack    {cm[1,0]:>6}  {cm[1,1]:>6}")
    
    def get_top_features(self, n: int = 15) -> pd.DataFrame:
        """Get top N most important features."""
        if self.feature_importance_ is None:
            raise RuntimeError("Feature importance not available")
        return self.feature_importance_.head(n)
    
    def save_model(self, path: Path) -> None:
        """Save trained model to disk."""
        if self.model is None:
            raise RuntimeError("No model to save")
        
        model_data = {
            'model': self.model,
            'model_type': self.model_type,
            'training_stats': self.training_stats_,
            'feature_importance': self.feature_importance_
        }
        
        joblib.dump(model_data, path)
        print(f"Model saved to: {path}")
    
    @classmethod
    def load_model(cls, path: Path) -> 'ModelTrainer':
        """Load trained model from disk."""
        model_data = joblib.load(path)
        
        instance = cls(model_type=model_data['model_type'])
        instance.model = model_data['model']
        instance.training_stats_ = model_data['training_stats']
        instance.feature_importance_ = model_data['feature_importance']
        
        return instance


class DataLoader:
    """Load preprocessed data for training."""
    
    @staticmethod
    def load_processed_data(
        data_dir: Path = Path("processed_data")
    ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """
        Load preprocessed training and testing data.
        
        Returns:
            Tuple of (X_train, y_train, X_test, y_test)
        """
        train_path = data_dir / "train_processed.csv"
        test_path = data_dir / "test_processed.csv"
        
        if not train_path.exists() or not test_path.exists():
            raise FileNotFoundError(
                "Preprocessed data not found. Run preprocessing pipeline first."
            )
        
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        
        # Separate features and target
        target_col = 'label'
        
        X_train = train_df.drop(columns=[target_col])
        y_train = train_df[target_col]
        
        X_test = test_df.drop(columns=[target_col])
        y_test = test_df[target_col]
        
        return X_train, y_train, X_test, y_test
