## Assignment: 2

## Shaheer Murtaza Shehri       512638
## Reem Saleha                  513059
## Sharjeel Farooq              520464

# Multiclass Intrusion Detection System (IDS)

Network intrusion detection pipeline built on the UNSW-NB15 dataset. Trains and evaluates multiple classifiers for multiclass attack categorization.

## Dataset

**Source:** UNSW-NB15 (4 CSV files, ~2.54M rows, 49 features)
Link to dataset: https://unsw-my.sharepoint.com/personal/z5025758_ad_unsw_edu_au/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fz5025758%5Fad%5Funsw%5Fedu%5Fau%2FDocuments%2FUNSW%2DNB15%20dataset%2FCSV%20Files&viewid=f8d1dec5%2Dcd5f%2D42ae%2D8b06%2D2fece580c74a


**Preprocessing steps:**
- Merges rare attack categories (Analysis, Backdoor, Shellcode, Worms) into Generic
- Strategic undersampling of majority classes to reduce imbalance
- Duplicate removal (final: 120,326 samples, 45 raw features)
- Feature engineering (domain features, correlation filtering, mutual information)
- Hybrid categorical encoding (target encoding for high-cardinality, one-hot for low-cardinality)
- Robust numerical scaling on wide-range columns
- BorderlineSMOTE on minority classes before training

**Final classes (6):** Normal, Generic, Exploits, Fuzzers, DoS, Reconnaissance

## Models

Two classifiers trained with class-balanced strategies:

| Model | Accuracy | F1 (macro) | F1 (weighted) | Precision (macro) | Recall (macro) |
|---|---|---|---|---|---|
| Random Forest | 0.9410 | 0.8806 | 0.9416 | 0.8771 | 0.8848 |
| XGBoost | 0.9443 | 0.8811 | 0.9439 | 0.8839 | 0.8793 |

**Random Forest:** 300 estimators, max_depth=25, balanced class weights, sqrt max features

**XGBoost:** 300 estimators, max_depth=8, lr=0.1, subsample=0.8, balanced sample weights

## Evaluation Metrics

Each model is evaluated with:
- Accuracy, Precision, Recall, F1 (both macro and weighted averages)
- Per-class precision, recall, F1, and quality labels (GOOD / OK / WEAK)
- Confusion matrix visualization
- Per-class F1 bar chart

Reports and figures are saved to `processed_data_mc/` and `processed_data_mc/figures/`.

## Error Analysis

Per-class breakdown highlights model weaknesses:

| Class | RF F1 | XGB F1 | Quality |
|---|---|---|---|
| Normal | 0.9935 | 0.9934 | GOOD |
| Generic | 0.9680 | 0.9720 | GOOD |
| Fuzzers | 0.9681 | 0.9747 | GOOD |
| Exploits | 0.8490 | 0.8590 | OK / GOOD |
| Reconnaissance | 0.9280 | 0.9255 | GOOD |
| DoS | 0.5773 | 0.5618 | WEAK |

DoS is the weakest class for both models due to low sample count (903 test samples). Exploits shows moderate confusion with other attack types. Confusion matrices are saved as PNG files for visual inspection.

## Model Serialization

Models are serialized using `joblib` and saved to `models/saved/`:

```
models/saved/
    improved_random_forest_mc.joblib
    improved_xgboost_mc.joblib
```

Each artifact contains: `model`, `class_labels`, `n_classes`, `trained_at`.

The fitted preprocessing pipeline is also saved:

```
processed_data_mc/combined_preprocessing_pipeline.joblib
```

## Reproducibility Controls

- Global `RANDOM_STATE = 42` used across train/test split, SMOTE, and all model initializers
- Stratified 80/20 train/test split
- Raw and preprocessed data splits saved to CSV for exact reproduction
- Pipeline configuration via `PreprocessingConfig` dataclass with fixed thresholds
- `default_mc_config()` provides a deterministic default configuration
- Deterministic pipeline verified by test (fitting twice produces identical output)

## Unit Tests (pytest)

46 tests across 8 test classes in `tests/test_pipeline.py`. All passing.

```
pytest tests/test_pipeline.py -v
```

| Test Class | Tests | Coverage |
|---|---|---|
| TestDataLoading | 9 | Raw data loading, CSV presence, feature catalog, saved splits |
| TestTargetEncoder | 3 | Fit/transform shape, unseen categories, fitted flag |
| TestCategoricalEncoder | 2 | Object column reduction, empty column handling |
| TestNumericalScaler | 4 | Robust scaling, skip columns, no-op strategy, fitted flag |
| TestFeatureEngineer | 3 | Fit state, correlated column removal, unfitted error |
| TestPreprocessingPipeline | 6 | Fit flag, transform output, NaN check, numeric check, stats, fit_transform |
| TestPredictionShape | 4 | Prediction length, proba shape, class range, proba sums |
| TestFeatureConsistency | 5 | Column match, count, dtypes, saved CSV consistency, target leakage |
| TestReproducibility | 4 | Pipeline determinism, transform idempotency, scaler/encoder reproducibility |
| TestConfig | 4 | Default config, multiclass config, invalid threshold/percentile |

## How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Train models and generate evaluation reports
python run.py

# Run tests
pytest tests/test_pipeline.py -v
```
