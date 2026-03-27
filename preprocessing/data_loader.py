import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, List, Union


class DataLoader:
    
    def __init__(self, data_dir: Union[str, Path] = "data"):
        self.data_dir = Path(data_dir)
        self.feature_names: List[str] = []
        self._validate_data_dir()
        
    def _validate_data_dir(self) -> None:
        # Validate that data directory exists and contains expected files.
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")

    def load_combined_data(self, nrows: Optional[int] = None) -> pd.DataFrame:
        # Load all UNSW-NB15 combined data files (UNSW-NB15_1.csv through UNSW-NB15_4.csv)
        files = [
            "UNSW-NB15_1.csv",
            "UNSW-NB15_2.csv",
            "UNSW-NB15_3.csv",
            "UNSW-NB15_4.csv"
        ]
        return self.load_full_data(nrows=nrows, file_names=files)
    
    def load_training_data(self) -> pd.DataFrame:
        path = self.data_dir / "UNSW_NB15_training-set.csv"
        if(not path.exists()):
            print(f"Training file not found: {path}, attempting to load from combined files")
            return self.load_combined_data()
        return self._load_csv(path, "Training")
    
    def load_testing_data(self) -> pd.DataFrame:
        path = self.data_dir / "UNSW_NB15_testing-set.csv"
        if(not path.exists()):
            print(f"Testing file not found: {path}, attempting to load from combined files")
            return self.load_combined_data()
        return self._load_csv(path, "Testing")
    
    def load_train_test(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        train_df = self.load_training_data()
        test_df = self.load_testing_data()
        return train_df, test_df
    
    def load_full_data(self, nrows: Optional[int] = None,file_names: List[str] = None) -> pd.DataFrame:
        main_files = file_names
        dfs = []
        
        for filename in main_files:
            path = self.data_dir / filename
            if path.exists():
                df = self._load_csv(path, filename, apply_feature_names=True, nrows=nrows)
                dfs.append(df)
            else:
                print(f"File not found: {path}")
        
        if not dfs:
            raise FileNotFoundError("No main data files found")

        combined = pd.concat(dfs, ignore_index=True)
        print(f"combined dataset shape: {combined.shape}")
        return combined
    
    def _load_csv(self, path: Path, name: str, apply_feature_names: bool = False, nrows: Optional[int] = None) -> pd.DataFrame:
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        print(f"loading {name} from {path}.")

        header = "infer"
        names = None
        dtype_map = None
        if apply_feature_names and self.feature_names:
            header = None
            names = self.feature_names
            dtype_map = self._build_dtype_map()

        # Force all columns to string on read to avoid dtype inference/casting errors
        df = pd.read_csv(
            path,
            low_memory=True,
            header=header,
            names=names,
            engine="c",
            on_bad_lines="warn",
            nrows=nrows,
        )
        df = self._clean_columns(df)
        print(f"  shape: {df.shape[0]:,} rows, {df.shape[1]} columns")
        return df

    def _clean_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        # Clean column names: strip whitespace and convert to lowercase with underscores.
        df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_", regex=False)
        return df

    def _build_dtype_map(self) -> Optional[dict]:
        if not self.feature_names:
            return None
        dtype_map = {name: "float32" for name in self.feature_names}
        for cat in ["proto", "service", "state", "attack_cat"]:
            if cat in dtype_map:
                dtype_map[cat] = "category"
        if "label" in dtype_map:
            dtype_map["label"] = "int8"
        return dtype_map

    def load_feature_catalog(self) -> Optional[pd.DataFrame]:
        path = self.data_dir / "NUSW-NB15_features.csv"
        if not path.exists():
            return None
        catalog = pd.read_csv(path, encoding="latin1", encoding_errors="ignore")
        catalog.columns = catalog.columns.str.strip()
        catalog["Name"] = catalog["Name"].str.strip()
        self.feature_names = catalog["Name"].str.replace(" ", "_", regex=False).tolist()
        return catalog
    
    def get_data_summary(self, df: pd.DataFrame) -> Dict:

        summary = {'shape': df.shape, 'columns': df.columns.tolist(), 'dtypes': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict(),
            'numerical_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_columns': df.select_dtypes(include=['object', 'string']).columns.tolist(),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
        }
        
        # Add target distribution if present
        if 'label' in df.columns:
            summary['label_distribution'] = df['label'].value_counts().to_dict()
        
        if 'attack_cat' in df.columns:
            summary['attack_category_distribution'] = df['attack_cat'].value_counts().to_dict()
        
        return summary
    
    def print_data_summary(self, df: pd.DataFrame) -> None:
        """Print formatted data summary to console."""
        summary = self.get_data_summary(df)
        
        print("\nDATASET SUMMARY\n")
        print(f"Shape: {summary['shape'][0]:,} rows x {summary['shape'][1]} columns")
        print(f"Memory Usage: {summary['memory_usage_mb']:.2f} MB")
        print(f"\nNumerical columns ({len(summary['numerical_columns'])})")
        print(f"Categorical columns ({len(summary['categorical_columns'])})")
        
        # Missing values
        missing_count = sum(1 for v in summary['missing_values'].values() if v > 0)
        if missing_count > 0:
            print(f"\nColumns with missing values: {missing_count}")
            for col, count in summary['missing_values'].items():
                if count > 0:
                    pct = summary['missing_percentage'][col]
                    print(f"  {col}: {count:,} ({pct:.2f}%)")
        else:
            print("\nNo missing values found!")
        
        # Target distribution
        if 'label_distribution' in summary:
            print("\nLabel Distribution:")
            for label, count in summary['label_distribution'].items():
                pct = count / sum(summary['label_distribution'].values()) * 100
                print(f"  {label}: {count:,} ({pct:.1f}%)")
        
        if 'attack_category_distribution' in summary:
            print("\nAttack Category Distribution:")
            for cat, count in sorted(summary['attack_category_distribution'].items(), 
                                    key=lambda x: x[1], reverse=True):
                pct = count / sum(summary['attack_category_distribution'].values()) * 100
                print(f"  {cat}: {count:,} ({pct:.1f}%)")
