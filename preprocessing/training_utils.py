from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from imblearn.over_sampling import SMOTE

from preprocessing import PreprocessingConfig, PreprocessingPipeline
from preprocessing.data_loader import DataLoader


MIN_ATTACK_COUNT = 9_000

# Strategic undersampling targets
UNDERSAMPLE_TARGETS = {
    'Normal': 50000,
    'Generic': 40000,
    'Exploits': 30000,
    'Fuzzers': None,  # Keep all
    'DoS': None,  # Keep all
    'Reconnaissance': None,  # Keep all
}


def default_mc_config(data_dir: Path, output_dir: Path) -> PreprocessingConfig:
    return PreprocessingConfig(
        data_dir=data_dir,
        output_dir=output_dir,
        target_column="target_mc",
        attack_category_column="attack_cat",
        imbalance_strategy="smote",
        encoding_strategy="hybrid",
        scaling_strategy="robust",
        columns_to_drop=sorted({"unnamed: 37", "unnamed: 38", "unnamed: 47", "srcip", "dstip"}),
        categorical_columns=["proto", "service", "state"],
        binary_columns=["is_sm_ips_ports", "is_ftp_login"],
        target_encoding_columns=["proto"],
        onehot_encoding_columns=["service", "state"],
        skip_scaling_columns=["is_ftp_login", "is_sm_ips_ports", "label", "target_mc"],
        correlation_threshold=0.90,
        mi_threshold=0.005,
        wide_range_threshold=1_000.0,
    )


def load_mc_dataset(data_dir: Path, sample_rows: int | None) -> Tuple[pd.DataFrame, pd.Series]:
    """Load dataset with improved class handling strategy"""
    loader = DataLoader(data_dir)
    loader.load_feature_catalog()
    df = loader.load_combined_data(nrows=sample_rows)
    df.columns = df.columns.str.strip().str.lower()
    if "label" not in df.columns:
        raise ValueError("label column missing")

    print("\nCleaning and normalizing categories...")
    df["label"] = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int)
    df["attack_cat"] = df.get("attack_cat", pd.Series(["Normal"] * len(df))).fillna("Normal")
    df["attack_cat"] = df["attack_cat"].str.strip()  # Remove leading/trailing spaces
    df["attack_cat"] = df["attack_cat"].replace({"-": "Normal", "normal": "Normal"})
    
    # Normalize category name variations
    category_mapping = {
        "Backdoors": "Backdoor",
        "backdoor": "Backdoor",
        "backdoors": "Backdoor",
        "exploits": "Exploits",
        "dos": "DoS",
        "reconnaissance": "Reconnaissance",
        "fuzzers": "Fuzzers",
        "shellcode": "Shellcode",
        "worms": "Worms",
        "analysis": "Analysis",
        "generic": "Generic"
    }
    df["attack_cat"] = df["attack_cat"].replace(category_mapping)
    
    print("Initial class distribution:")
    for cat, count in df['attack_cat'].value_counts().items():
        pct = 100 * count / len(df)
        print(f"  {cat:20s} {count:>10,} ({pct:>6.2f}%)")
    
    # Step 2: Merge rare classes into Generic (BEFORE undersampling)
    print(f"\nMerging rare classes into Generic (threshold: {MIN_ATTACK_COUNT:,}).")
    attack_counts = df['attack_cat'].value_counts()
    rare_attacks = [
        cat for cat, count in attack_counts.items() 
        if count < MIN_ATTACK_COUNT and cat not in ['Normal', 'Generic']
    ]
    
    if rare_attacks:
        print("Rare categories to merge into Generic:")
        rare_total = 0
        for cat in rare_attacks:
            count = attack_counts[cat]
            rare_total += count
            print(f"  {cat:20s} {count:>10,}")
        print(f"  {'-'*20} {rare_total:>10,}")
        df.loc[df["attack_cat"].isin(rare_attacks), "attack_cat"] = "Generic"
        print(f"Merged {len(rare_attacks)} rare categories into Generic")
    else:
        print("No rare categories found (all have sufficient samples)")
    
    print("\nAfter merging:")
    for cat, count in df['attack_cat'].value_counts().items():
        pct = 100 * count / len(df)
        print(f"  {cat:20s} {count:>10,} ({pct:>6.2f}%)")
    
    # Step 3: Strategic undersampling
    print("\n Strategic undersampling.")
    sampled_dfs = []
    for cat in df['attack_cat'].unique():
        cat_df = df[df['attack_cat'] == cat]
        current_size = len(cat_df)
        target_size = UNDERSAMPLE_TARGETS.get(cat, None)
        
        if target_size and current_size > target_size:
            cat_df = cat_df.sample(n=target_size, random_state=42)
            print(f"  {cat:20s} {current_size:>10,} -> {target_size:>10,} (undersampled)")
        else:
            print(f"  {cat:20s} {current_size:>10,} -> {current_size:>10,} (kept all)")
        
        sampled_dfs.append(cat_df)
    
    df = pd.concat(sampled_dfs, ignore_index=True)
    
    print(f"\nDataset size after undersampling: {len(df):,}")
    print("Final distribution:")
    for cat, count in df['attack_cat'].value_counts().items():
        pct = 100 * count / len(df)
        print(f"  {cat:20s} {count:>10,} ({pct:>6.2f}%)")

    # Create target
    df["target_mc"] = df["attack_cat"]
    df.loc[df["label"] == 0, "target_mc"] = "Normal"

    y = df["target_mc"].astype(str)
    X = df.drop(columns=["target_mc", "attack_cat", "label", "srcip", "dstip"], errors="ignore")

    # Shuffle and remove duplicates
    X, y = shuffle(X, y, random_state=42)
    dup_mask = X.duplicated()
    if dup_mask.any():
        X = X.loc[~dup_mask].reset_index(drop=True)
        y = y.loc[~dup_mask].reset_index(drop=True)
        print(f"\nRemoved {dup_mask.sum():,} duplicate rows; final size: {len(X):,}")

    return X, y


def split_mc(X: pd.DataFrame, y: pd.Series, seed: int = 42):
    vc = y.value_counts()
    rare = vc[vc < 3].index.tolist()
    if rare:
        print(f"Merging rare classes (<3 samples) into RARE: {rare}")
    y_strat = y.where(~y.isin(rare), "RARE")
    if y_strat.value_counts().min() < 2:
        keep_mask = y_strat != "RARE"
        removed = int((~keep_mask).sum())
        X = X.loc[keep_mask].reset_index(drop=True)
        y = y.loc[keep_mask].reset_index(drop=True)
        y_strat = y_strat.loc[keep_mask].reset_index(drop=True)
        print(f"Dropped {removed} ultra-rare rows (<2) to enable stratification")
    strat_base = y_strat if y_strat.value_counts().min() >= 2 else None
    X_train, X_tmp, y_train, y_tmp, y_strat_train, y_strat_tmp = train_test_split(
        X,
        y,
        y_strat,
        test_size=0.6,
        stratify=strat_base,
        random_state=seed,
        shuffle=True,
    )

    strat_tmp = y_strat_tmp if y_strat_tmp.value_counts().min() >= 2 else None
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp,
        y_tmp,
        test_size=2 / 3,
        stratify=strat_tmp,
        random_state=seed,
        shuffle=True,
    )
    print(
        f"Splits: train={len(X_train):,}, val={len(X_val):,}, test={len(X_test):,}; "
        f"classes={y.nunique()}"
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def fit_preprocess_mc(X_train, y_train, X_val, X_test, config: PreprocessingConfig):
    pipe = PreprocessingPipeline(config)
    label_codes = y_train.astype("category").cat.codes
    pipe.fit(X_train, label_codes)
    X_train_proc = pipe.transform(X_train)
    X_val_proc = pipe.transform(X_val)
    X_test_proc = pipe.transform(X_test)
    return pipe, X_train_proc, y_train, X_val_proc, X_test_proc


def balance_train(X_train_proc, y_train_proc, apply_smote=False):
    """Balance training data - SMOTE optional"""
    counts = y_train_proc.value_counts()
    min_count = counts.min()
    
    if not apply_smote:
        print("\nSkipping SMOTE - using class_weight='balanced' instead")
        print("Class distribution in training set:")
        for cls, cnt in counts.items():
            print(f"  {cls:20s} {cnt:>10,}")
        return X_train_proc, y_train_proc
    
    if min_count < 2:
        print(f"Cannot apply SMOTE: min class count {min_count} < 2")
        print("Using original training set with class weights")
        return X_train_proc, y_train_proc

    print("\nApplying SMOTE (not recommended - use class weights instead).")
    k_neighbors = min(5, min_count - 1)
    smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
    Xb, yb = smote.fit_resample(X_train_proc, y_train_proc)
    print(f"Dataset: {len(X_train_proc):,} -> {len(Xb):,} rows")
    return Xb, yb


def train_rf(X_train, y_train) -> RandomForestClassifier:
    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=42,
    )
    clf.fit(X_train, y_train)
    return clf


def evaluate_split(model, X, y, split: str):
    preds = model.predict(X)
    acc = accuracy_score(y, preds)
    report = classification_report(y, preds, output_dict=True)
    cm = confusion_matrix(y, preds).tolist()
    print(f"{split} accuracy: {acc:.4f}")
    return {"accuracy": acc, "report": report, "confusion_matrix": cm}


def save_artifacts(pipe, model, results, output_dir: Path, model_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, output_dir / "preprocessing_pipeline_mc.joblib")
    joblib.dump(model, model_dir / "random_forest_mc.joblib")
    with (output_dir / "mc_results.json").open("w") as f:
        json.dump(results, f, indent=2)
