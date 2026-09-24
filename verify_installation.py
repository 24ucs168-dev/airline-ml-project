import sys
from src.data_preprocessing import load_dataset, inspect_dataset
from src.train_model import run_full_training
from src.predict import predict_sample

print("1. Loading dataset...")
df, path = load_dataset()
print(f"Loaded {df.shape} from {path}")

print("2. Inspecting dataset...")
info = inspect_dataset(df)
print(f"Target: {info['target_column']}, Task: {info['task_type']}")
print(f"Numeric features: {len(info['numeric_features'])}")
print(f"Categorical features: {len(info['categorical_features'])}")
print(f"Missing values: {info['missing_values']}")

print("3. Training model (sample of 40,000 for rapid artifact creation)...")
res = run_full_training(df, sample_size=40000)
print("Trained successfully!")
print("Accuracy:", res["evaluation"]["accuracy"])
print("F1:", res["evaluation"]["f1_score"])
print("Top 5 features:", [f["feature"] for f in res["evaluation"]["top_features"][:5]])

print("4. Testing prediction on first test sample...")
sample = res["X_test"].iloc[0].to_dict()
actual = res["y_test"].iloc[0]
pred = predict_sample(res["pipeline"], sample)
print("Prediction result:", pred)
print("Actual:", actual)
