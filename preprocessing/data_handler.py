from pathlib import Path
from typing import Tuple, Dict, Any
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import joblib

from preprocessing.training_utils import load_mc_dataset, default_mc_config
from preprocessing.pipeline import PreprocessingPipeline


class DataHandler:

    def __init__(self, data_dir: Path = Path("data"), output_dir: Path = Path("processed_data")):

        self.data_dir = data_dir
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Data containers
        self.X_original = None
        self.y_original = None
        self.class_labels = None
        self.y_numeric = None
        
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        self.X_train_prep = None
        self.X_test_prep = None
        
        self.pipeline = None
        
    def load_raw_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Load raw data"""
        print("\nLoading raw data.")
        self.X_original, self.y_original = load_mc_dataset(self.data_dir, sample_rows=None)
        
        print(f"Loaded data: {self.X_original.shape[0]:,} samples, {self.X_original.shape[1]} features")
        print(f"Classes: {self.y_original.nunique()} unique")
        
        return self.X_original, self.y_original
    
    def encode_target(self) -> Tuple[pd.Series, np.ndarray]:

        print("\nEncoding target labels...")
        
        self.y_numeric, self.class_labels = pd.factorize(self.y_original)
        self.y_numeric = pd.Series(self.y_numeric, name="target_mc")
        
        print(f"Class mapping:")
        for idx, label in enumerate(self.class_labels):
            count = (self.y_numeric == idx).sum()
            print(f"  {idx}: {label:20s} ({count:,} samples)")
        
        return self.y_numeric, self.class_labels
    
    def split_data(self,test_size: float = 0.2,
        random_state: int = 42,save: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:

        print("\nSplitting data (BEFORE preprocessing).")
        
        if self.y_numeric is None:
            raise ValueError("Must encode target first. Call encode_target().")
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X_original,
            self.y_numeric,
            test_size=test_size,
            random_state=random_state,
            stratify=self.y_numeric
        )
        
        print(f"Training set: {len(self.X_train):,} samples ({100*(1-test_size):.0f}%)")
        print(f"Test set: {len(self.X_test):,} samples ({100*test_size:.0f}%)")
        
        # Save splits if requested
        if save:
            self._save_data_splits()
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def _save_data_splits(self) -> None:
        """Save train/test splits to CSV files"""
        
        print("\nSaving data splits...")
        
        # Create split directory
        splits_dir = self.output_dir / "data_splits"
        splits_dir.mkdir(parents=True, exist_ok=True)
        
        # Save training set
        X_train_with_target = pd.concat(
            [self.X_train, self.y_train.rename("target")],
            axis=1
        )
        train_path = splits_dir / "X_train_raw.csv"
        X_train_with_target.to_csv(train_path, index=False)
        print(f"  Training set saved: {train_path}")
        
        # Save test set
        X_test_with_target = pd.concat(
            [self.X_test, self.y_test.rename("target")],
            axis=1
        )
        test_path = splits_dir / "X_test_raw.csv"
        X_test_with_target.to_csv(test_path, index=False)
        print(f"  Test set saved: {test_path}")
        
        # Save metadata
        metadata = {
            'train_size': len(self.X_train),
            'test_size': len(self.X_test),
            'num_features': self.X_train.shape[1],
            'num_classes': len(self.class_labels),
            'class_labels': list(self.class_labels),
            'test_size_ratio': 0.2,
            'random_state': 42
        }
        
        metadata_path = splits_dir / "split_metadata.joblib"
        joblib.dump(metadata, metadata_path)
        print(f"  Metadata saved: {metadata_path}")
    
    def preprocess_data(self, apply_imbalance_handling: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        print("\nPreprocessing.")
        
        if self.X_train is None or self.X_test is None:
            raise ValueError("Must split data first. Call split_data().")
        
        # Create pipeline
        config = default_mc_config(
            data_dir=self.data_dir,
            output_dir=self.output_dir
        )
        config.imbalance_strategy = "none"  # Use class weights instead
        
        self.pipeline = PreprocessingPipeline(config)
        
        # Fit on training data ONLY
        print("  Fitting pipeline on training data only...")
        self.X_train_prep, self.y_train_prep = self.pipeline.fit_transform(
            self.X_train,
            self.y_train,
            apply_imbalance_handling=apply_imbalance_handling
        )
        
        # Transform test data with fitted pipeline
        print("  Transforming test data...")
        self.X_test_prep = self.pipeline.transform(self.X_test)
        
        print(f"  Preprocessed features: {self.X_train_prep.shape[1]}")
        print(f"  Training set (preprocessed): {len(self.X_train_prep):,} samples")
        print(f"  Test set (preprocessed): {len(self.X_test_prep):,} samples")
        
        return self.X_train_prep, self.X_test_prep
    
    def save_preprocessed_data(self) -> None:
        """Save preprocessed train/test splits"""
        
        print("\nSaving preprocessed data.")
        
        if self.X_train_prep is None or self.X_test_prep is None:
            print("No preprocessed data available. Call preprocess_data() first.")
            return
        
        # Create preprocessed directory
        prep_dir = self.output_dir / "data_splits"
        prep_dir.mkdir(parents=True, exist_ok=True)
        
        # Save preprocessed training set (features + target together)
        X_train_prep_df = pd.DataFrame(self.X_train_prep)
        X_train_prep_with_target = pd.concat(
            [X_train_prep_df, self.y_train.reset_index(drop=True).rename("target")],
            axis=1
        )
        train_prep_path = prep_dir / "X_train_preprocessed.csv"
        X_train_prep_with_target.to_csv(train_prep_path, index=False)
        print(f"  Preprocessed training set saved: {train_prep_path}")
        
        # Save preprocessed test set (features + target together)
        X_test_prep_df = pd.DataFrame(self.X_test_prep)
        X_test_prep_with_target = pd.concat(
            [X_test_prep_df, self.y_test.reset_index(drop=True).rename("target")],
            axis=1
        )
        test_prep_path = prep_dir / "X_test_preprocessed.csv"
        X_test_prep_with_target.to_csv(test_prep_path, index=False)
        print(f"  Preprocessed test set saved: {test_prep_path}")
    
    def save_pipeline(self) -> None:
        """Save preprocessing pipeline"""
        
        if self.pipeline is None:
            print("No pipeline available. Call preprocess_data() first.")
            return
        
        print("\nSaving preprocessing pipeline...")
        
        pipeline_path = self.output_dir / "improved_preprocessing_pipeline.joblib"
        joblib.dump(self.pipeline, pipeline_path)
        print(f"  Pipeline saved: {pipeline_path}")
    
    def get_data(self) -> Dict[str, Any]:

        return {
            'X_train': self.X_train_prep,
            'X_test': self.X_test_prep,
            'y_train': self.y_train,
            'y_test': self.y_test,
            'class_labels': self.class_labels,
            'y_numeric': self.y_numeric,
            'X_train_raw': self.X_train,
            'X_test_raw': self.X_test
        }


def prepare_data_pipeline(data_dir: Path = Path("data"),
    output_dir: Path = Path("processed_data_mc"),
    test_size: float = 0.2) -> Dict[str, Any]:

    print("DATA PREPARATION PIPELINE")
    
    # Initialize handler
    handler = DataHandler(data_dir, output_dir)
    
    # Load raw data
    handler.load_raw_data()
    
    # Encode target
    handler.encode_target()
    
    # Split data (BEFORE preprocessing)
    handler.split_data(test_size=test_size, save=True)
    
    # Preprocess (AFTER split)
    handler.preprocess_data(apply_imbalance_handling=False)
    
    # Save preprocessed data
    handler.save_preprocessed_data()
    
    # Save pipeline
    handler.save_pipeline()
    
    print("\n" + "="*70)
    print("DATA PREPARATION COMPLETED")
    print("="*70)
    
    return handler.get_data()
