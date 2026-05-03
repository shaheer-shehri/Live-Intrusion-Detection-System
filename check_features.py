import joblib
import pickle

print("="*60)
print("CHECKING PIPELINE OUTPUT FEATURES")
print("="*60)

try:
    pipeline = joblib.load('processed_data_mc/combined_preprocessing_pipeline.joblib')
    feature_names = pipeline.get_feature_names()
    print(f'\nPipeline output features ({len(feature_names)}):')
    for i, name in enumerate(feature_names):
        print(f"  {i}: {name}")
except Exception as e:
    print(f'Error getting feature names: {type(e).__name__}: {e}')

print("\n" + "="*60)
print("CHECKING MODEL INPUT FEATURES")
print("="*60)

try:
    model = pickle.load(open('models/saved/improved_xgboost_mc.joblib', 'rb'))
    print(f'\nModel n_features_in_: {model.n_features_in_}')
    if hasattr(model, 'get_booster'):
        print(f'Model booster num_features: {model.get_booster().num_features()}')
except Exception as e:
    print(f'Error loading model: {type(e).__name__}: {e}')

print("\n" + "="*60)
print("CHECKING TEST DATA")
print("="*60)

try:
    import pandas as pd
    df = pd.read_csv('processed_data_mc/data_splits/X_test_raw.csv')
    print(f'\nX_test_raw.csv shape: {df.shape}')
    print(f'Columns ({len(df.columns)}): {df.columns.tolist()}')
except Exception as e:
    print(f'Error reading test data: {type(e).__name__}: {e}')
