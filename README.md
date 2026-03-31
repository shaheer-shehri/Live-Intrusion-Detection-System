## Assignment: 3

## Shaheer Murtaza Shehri       512638
## Reem Saleha                  513059
## Muhammad Sharjeel Farooq     520464

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

## API Tests

Run the FastAPI endpoint tests (uses mocked model/pipeline):

```bash
pytest tests/test_api.py -s
```


## Predict via API with real model

Start the API (artifacts required):

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Send a prediction request with all 45 features (proto/service/state as strings; srcip/dstip can be empty):

```bash
curl -X POST http://127.0.0.1:8000/predict \
    -H "Content-Type: application/json" \
    -d '{
        "srcip": "",
        "dstip": "",
        "proto": "tcp",
        "service": "http",
        "state": "est",
        "sport": 12345,
        "dsport": 80,
        "dur": 1.2,
        "sbytes": 500,
        "dbytes": 1200,
        "sttl": 60,
        "dttl": 60,
        "sloss": 0,
        "dloss": 0,
        "sload": 10,
        "dload": 20,
        "spkts": 8,
        "dpkts": 10,
        "swin": 30000,
        "dwin": 30000,
        "stcpb": 1000,
        "dtcpb": 2000,
        "smeansz": 62.5,
        "dmeansz": 120,
        "trans_depth": 1,
        "res_bdy_len": 0,
        "sjit": 0.05,
        "djit": 0.04,
        "stime": 0,
        "ltime": 1.2,
        "sintpkt": 0.15,
        "dintpkt": 0.12,
        "tcprtt": 0.02,
        "synack": 0.01,
        "ackdat": 0.01,
        "is_sm_ips_ports": 0,
        "ct_state_ttl": 3,
        "ct_flw_http_mthd": 1,
        "is_ftp_login": 0,
        "ct_ftp_cmd": 0,
        "ct_srv_src": 2,
        "ct_srv_dst": 2,
        "ct_dst_ltm": 4,
        "ct_src__ltm": 3,
        "ct_src_dport_ltm": 2,
        "ct_dst_sport_ltm": 2,
        "ct_dst_src_ltm": 3
    }'
```

Windows cmd version (uses `^` for line breaks and doubled quotes inside JSON):

```cmd
curl -X POST http://127.0.0.1:8000/predict ^
    -H "Content-Type: application/json" ^
    -d "{""srcip"":"""",""dstip"":"""",""proto"":""tcp"",""service"":""http"",""state"":""est"",""sport"":12345,""dsport"":80,""dur"":1.2,""sbytes"":500,""dbytes"":1200,""sttl"":60,""dttl"":60,""sloss"":0,""dloss"":0,""sload"":10,""dload"":20,""spkts"":8,""dpkts"":10,""swin"":30000,""dwin"":30000,""stcpb"":1000,""dtcpb"":2000,""smeansz"":62.5,""dmeansz"":120,""trans_depth"":1,""res_bdy_len"":0,""sjit"":0.05,""djit"":0.04,""stime"":0,""ltime"":1.2,""sintpkt"":0.15,""dintpkt"":0.12,""tcprtt"":0.02,""synack"":0.01,""ackdat"":0.01,""is_sm_ips_ports"":0,""ct_state_ttl"":3,""ct_flw_http_mthd"":1,""is_ftp_login"":0,""ct_ftp_cmd"":0,""ct_srv_src"":2,""ct_srv_dst"":2,""ct_dst_ltm"":4,""ct_src__ltm"":3,""ct_src_dport_ltm"":2,""ct_dst_sport_ltm"":2,""ct_dst_src_ltm"":3}"
```
## Stress Handling:
Stress management includes:
Rate Limiting:
    - Uses a sliding window algorithm to track requests per IP
    - Each IP gets a configurable number of requests per time window
    - Exceeding the limit returns HTTP 429 (Too Many Requests)

Circuit Breaker:
    - Monitors failure count within a time window
    - States: CLOSED (normal), OPEN (failing fast), HALF_OPEN (testing recovery)
    - When failures exceed threshold, circuit OPENS and rejects requests
    - After cooldown, moves to HALF_OPEN to test if service recovered

Metrics:
    - request_count: Total requests received
    - request_latency_seconds: Histogram of response times
    - active_requests: Current concurrent requests
    - error_count: Total errors by type

## Stress Testing:
Defined load testing scenarios using Locust:
Run with: locust -f stress_tests/locustfile.py --host=http://localhost:8000

HOW LOCUST WORKS:
1. Locust spawns virtual "users" that simulate real API clients
2. Each user executes tasks defined in the User class
3. Tasks are weighted by the @task decorator (higher = more frequent)
4. Locust measures:
   - Response Time: How long each request takes
   - Throughput: Requests per second (RPS)
   - Failure Rate: Percentage of failed requests

RUNNING STRESS TESTS:
---------------------
#### Basic test (10 users, spawn 1/sec)
locust -f stress_tests/locustfile.py --host=http://localhost:8000 \\
    --users 10 --spawn-rate 1 --run-time 60s --headless

#### Heavy load test (100 users)
locust -f stress_tests/locustfile.py --host=http://localhost:8000 \\
    --users 100 --spawn-rate 10 --run-time 120s --headless

#### Web UI mode (interactive)
locust -f stress_tests/locustfile.py --host=http://localhost:8000
#### Then open http://localhost:8089

#### Generate HTML report
locust -f stress_tests/locustfile.py --host=http://localhost:8000 \\
    --users 50 --spawn-rate 5 --run-time 60s --headless --html=report.html
