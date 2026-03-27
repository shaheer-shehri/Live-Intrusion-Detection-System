from pathlib import Path
import json
import pandas as pd
import joblib
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier
from imblearn.over_sampling import BorderlineSMOTE

from preprocessing.training_utils import load_mc_dataset, default_mc_config
from preprocessing.pipeline import PreprocessingPipeline
from models.evaluation import (
    evaluate_model,
    generate_evaluation_report,
    compare_models,
)


DATA_DIR   = Path("data")
OUTPUT_DIR = Path("processed_data_mc")
MODEL_DIR  = Path("models/saved")

SMOTE_MIN_TARGET = 12_000   # Boost small classes up to this many samples
SMOTE_K          = 5
RANDOM_STATE     = 42


def _balanced_smote(X_train, y_train):
    counts = y_train.value_counts()
    print("\n  Training class distribution BEFORE SMOTE:")
    for cls in sorted(counts.index):
        print(f"    {cls:>3d}  {counts[cls]:>8,}")

    strategy = {}
    for cls, cnt in counts.items():
        if cnt < SMOTE_MIN_TARGET:
            strategy[cls] = SMOTE_MIN_TARGET
    if not strategy:
        print("  All classes above SMOTE target – skipping SMOTE")
        return X_train, y_train

    min_k = min(SMOTE_K, counts.min() - 1)
    min_k = max(min_k, 1)

    print(f"\n  Applying BorderlineSMOTE (k={min_k}) ...")
    for cls, tgt in strategy.items():
        print(f"    class {cls}: {counts[cls]:,} -> {tgt:,}")

    smote = BorderlineSMOTE(
        sampling_strategy=strategy,
        k_neighbors=min_k,
        random_state=RANDOM_STATE,
    )
    X_res, y_res = smote.fit_resample(X_train, y_train)

    print(f"  Dataset after SMOTE: {len(X_res):,} samples")
    new_counts = pd.Series(y_res).value_counts()
    for cls in sorted(new_counts.index):
        print(f"    {cls:>3d}  {new_counts[cls]:>8,}")
    return X_res, y_res


def main():
    print("  IMPROVED MULTICLASS IDS TRAINING PIPELINE")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    figures_dir = OUTPUT_DIR / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    splits_dir = OUTPUT_DIR / "data_splits"
    splits_dir.mkdir(parents=True, exist_ok=True)


    print("\n Load & prepare dataset ")
    X, y_str = load_mc_dataset(DATA_DIR, sample_rows=None)
    print(f"Loaded {len(X):,} samples, {X.shape[1]} features")
    print(f"Classes: {sorted(y_str.unique())}")

    print("\nEncode target labels ")
    y_numeric, class_labels = pd.factorize(y_str)
    y_numeric = pd.Series(y_numeric, name="target_mc", index=X.index)

    print("\nClass mapping:")
    for idx, label in enumerate(class_labels):
        cnt = (y_numeric == idx).sum()
        pct = 100 * cnt / len(y_numeric)
        print(f"  {idx}: {label:20s} ({cnt:>7,} samples, {pct:.2f}%)")


    print("\nTrain / test split (80/20, stratified)")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_numeric,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y_numeric,
    )
    print(f"  Train: {len(X_train):,}   Test: {len(X_test):,}")

    # Save raw splits
    pd.concat([X_train, y_train.rename("target")], axis=1).to_csv(
        splits_dir / "X_train_raw.csv", index=False)
    pd.concat([X_test, y_test.rename("target")], axis=1).to_csv(
        splits_dir / "X_test_raw.csv", index=False)
    print("  Raw splits saved.")

    print("\nPreprocessing (fit on train)")
    config = default_mc_config(DATA_DIR, OUTPUT_DIR)
    pipeline = PreprocessingPipeline(config)

    X_train_prep, y_train_prep = pipeline.fit_transform(
        X_train, y_train, apply_imbalance_handling=False)
    X_test_prep = pipeline.transform(X_test)

    print(f"  Train preprocessed: {X_train_prep.shape}")
    print(f"  Test  preprocessed: {X_test_prep.shape}")

    # Save preprocessed splits (features + target together)
    pd.concat([X_train_prep, y_train.reset_index(drop=True).rename("target")], axis=1).to_csv(
        splits_dir / "X_train_preprocessed.csv", index=False)
    pd.concat([X_test_prep, y_test.reset_index(drop=True).rename("target")], axis=1).to_csv(
        splits_dir / "X_test_preprocessed.csv", index=False)
    print("  Preprocessed splits saved.")

    joblib.dump(pipeline, OUTPUT_DIR / "combined_preprocessing_pipeline.joblib")


    print("\nBalance training set (BorderlineSMOTE) ")
    X_train_bal, y_train_bal = _balanced_smote(
        X_train_prep if isinstance(X_train_prep, pd.DataFrame)
        else pd.DataFrame(X_train_prep),
        y_train_prep,
    )

    print("\nTrain models ")

    print("\n 1. Random Forest .")
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=25,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        class_weight='balanced',
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    rf.fit(X_train_bal, y_train_bal)
    print("        Random Forest trained.")

    print("\n 2 XGBoost .")
    sw = compute_sample_weight('balanced', y_train_bal)
    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        eval_metric='mlogloss',
        use_label_encoder=False,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbosity=0,
    )
    xgb.fit(X_train_bal, y_train_bal, sample_weight=sw)
    print("        XGBoost trained.")

    print("\nEvaluation")

    evaluations = []
    for name, model in [("Random Forest", rf), ("XGBoost", xgb)]:
        ev = evaluate_model(
            model=model,
            X_test=X_test_prep,
            y_test=y_test,
            class_labels=class_labels,
            model_name=name,
        )
        evaluations.append(ev)
        generate_evaluation_report(ev, output_dir=OUTPUT_DIR, model_name=name)

    print("\n Model comparison")
    compare_models(evaluations, output_dir=OUTPUT_DIR)


    print("\nSave artefacts")

    for tag, model in [("random_forest", rf), ("xgboost", xgb)]:
        path = MODEL_DIR / f"improved_{tag}_mc.joblib"
        joblib.dump({
            "model": model,
            "class_labels": list(class_labels),
            "n_classes": len(class_labels),
            "trained_at": datetime.now().isoformat(),
        }, path)
        print(f"  Saved {path}")

    # Save combined metrics
    combined = {ev.model_name: ev.get_metrics_summary() for ev in evaluations}
    with open(OUTPUT_DIR / "combined_eval_metrics.json", "w") as f:
        json.dump(combined, f, indent=2)
    print(f"  Combined metrics saved to {OUTPUT_DIR / 'combined_eval_metrics.json'}")

    print("  PIPELINE COMPLETED SUCCESSFULLY")
    print(f"  Models:  {MODEL_DIR}")
    print(f"  Results: {OUTPUT_DIR}")
    print(f"  Figures: {figures_dir}")


if __name__ == "__main__":
    main()
