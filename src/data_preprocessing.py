"""
Data Preprocessing Module
Handles automatic dataset discovery, inspection, cleaning, feature identification,
and scikit-learn preprocessing pipeline construction.
"""

import os
import glob
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def find_dataset_path(base_dir: Optional[str] = None) -> Optional[str]:
    """
    Automatically search for and return the primary dataset path in the workspace.
    Prioritizes datasets in data/ or the root directory.
    """
    if base_dir is None:
        base_dir = os.getcwd()
    
    base_path = Path(base_dir)
    
    # Priority 1: Specifically known dataset
    specific_candidates = [
        base_path / "data" / "Airline_customer_satisfaction.csv",
        base_path / "Airline_customer_satisfaction.csv"
    ]
    for candidate in specific_candidates:
        if candidate.exists():
            return str(candidate.resolve())

    # Priority 2: Any .csv in data/ or root
    search_dirs = [base_path / "data", base_path]
    for sdir in search_dirs:
        if sdir.exists():
            csv_files = list(sdir.glob("*.csv"))
            if csv_files:
                # Return the largest CSV file (most likely the primary dataset)
                return str(max(csv_files, key=lambda f: f.stat().st_size).resolve())
            
            # Check for parquet or excel
            other_files = list(sdir.glob("*.xlsx")) + list(sdir.glob("*.parquet"))
            if other_files:
                return str(max(other_files, key=lambda f: f.stat().st_size).resolve())

    return None


def load_dataset(file_path: Optional[str] = None) -> Tuple[pd.DataFrame, str]:
    """
    Loads dataset from file_path or automatically detected path.
    Returns (DataFrame, file_path_used).
    """
    if file_path is None or not os.path.exists(file_path):
        detected = find_dataset_path()
        if not detected:
            raise FileNotFoundError("No dataset found in project folder or 'data/' directory.")
        file_path = detected

    ext = Path(file_path).suffix.lower()
    if ext == ".csv":
        df = pd.read_csv(file_path)
    elif ext in [".xls", ".xlsx"]:
        df = pd.read_excel(file_path)
    elif ext == ".parquet":
        df = pd.read_parquet(file_path)
    else:
        # Default try CSV
        df = pd.read_csv(file_path)

    return df, file_path


def detect_target_column(df: pd.DataFrame, candidate: Optional[str] = None) -> str:
    """
    Automatically detects the target column from a DataFrame.
    """
    if candidate and candidate in df.columns:
        return candidate

    # Common target column name patterns
    target_keywords = [
        "satisfaction", "target", "label", "outcome", "class", "churn",
        "status", "response", "result", "survived", "default", "price"
    ]
    
    col_names_lower = {col.lower().strip(): col for col in df.columns}
    for kw in target_keywords:
        for col_lower, original_name in col_names_lower.items():
            if kw == col_lower or kw in col_lower:
                return original_name

    # Check for categorical or binary columns near the end of the dataframe
    for col in reversed(df.columns):
        if df[col].nunique() in [2, 3]:
            return col

    # Fallback to the last column
    return df.columns[-1]


def detect_task_type(series: pd.Series) -> str:
    """
    Determines whether the task is 'classification' or 'regression'.
    """
    # If categorical / object / boolean dtype
    if series.dtype == "object" or series.dtype.name in ["category", "bool"]:
        return "classification"
    
    # If numeric, check unique count
    n_unique = series.nunique()
    if n_unique <= 20:
        return "classification"
    
    return "regression"


def identify_columns(df: pd.DataFrame, target_col: str) -> Tuple[List[str], List[str], List[str]]:
    """
    Identifies numeric features, categorical features, and dropped ID/uninformative columns.
    Returns (numeric_cols, categorical_cols, dropped_cols).
    """
    features_df = df.drop(columns=[target_col], errors="ignore")
    numeric_cols: List[str] = []
    categorical_cols: List[str] = []
    dropped_cols: List[str] = []

    for col in features_df.columns:
        col_lower = col.lower().strip()
        n_unique = features_df[col].nunique()
        total_len = len(features_df)

        # Check for potential ID columns (e.g. 'id', 'passenger_id', 'index', or 100% unique strings)
        is_id_name = (col_lower in ["id", "index", "uid", "guid"] or 
                      col_lower.endswith("_id") or 
                      col_lower.endswith("id"))
        
        # If it's named like an ID and unique ratio is high, drop it
        if is_id_name and (n_unique / total_len > 0.85):
            dropped_cols.append(col)
            continue
            
        # Single value constant columns provide zero information
        if n_unique <= 1:
            dropped_cols.append(col)
            continue

        if pd.api.types.is_numeric_dtype(features_df[col]):
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)

    return numeric_cols, categorical_cols, dropped_cols


def inspect_dataset(df: pd.DataFrame, target_col: Optional[str] = None) -> Dict[str, Any]:
    """
    Computes comprehensive structural and statistical information about the dataset.
    """
    detected_target = detect_target_column(df, target_col)
    task_type = detect_task_type(df[detected_target])
    numeric_cols, categorical_cols, dropped_cols = identify_columns(df, detected_target)

    # Missing value analysis
    null_counts = df.isnull().sum()
    null_dict = {
        col: {
            "count": int(count),
            "percentage": round(float((count / len(df)) * 100), 2)
        }
        for col, count in null_counts.items() if count > 0
    }

    # Class distribution if classification
    class_dist = {}
    if task_type == "classification":
        counts = df[detected_target].value_counts(dropna=False)
        percentages = (df[detected_target].value_counts(normalize=True, dropna=False) * 100).round(2)
        for label, count in counts.items():
            class_dist[str(label)] = {
                "count": int(count),
                "percentage": float(percentages[label])
            }

    # Dtypes
    dtypes_dict = {col: str(dtype) for col, dtype in df.dtypes.items()}

    return {
        "shape": df.shape,
        "total_rows": int(df.shape[0]),
        "total_cols": int(df.shape[1]),
        "target_column": detected_target,
        "task_type": task_type,
        "numeric_features": numeric_cols,
        "categorical_features": categorical_cols,
        "dropped_features": dropped_cols,
        "missing_values": null_dict,
        "total_missing_cells": int(null_counts.sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "class_distribution": class_dist,
        "dtypes": dtypes_dict,
        "columns": list(df.columns)
    }


def build_preprocessor(numeric_features: List[str], categorical_features: List[str]) -> ColumnTransformer:
    """
    Creates an sklearn ColumnTransformer with appropriate imputation and encoding.
    """
    transformers = []

    if numeric_features:
        num_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
        ])
        transformers.append(("num", num_transformer, numeric_features))

    if categorical_features:
        cat_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])
        transformers.append(("cat", cat_transformer, categorical_features))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    return preprocessor


def prepare_data(
    df: pd.DataFrame,
    target_col: Optional[str] = None,
    test_size: float = 0.2,
    random_state: int = 42,
    sample_size: Optional[int] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, ColumnTransformer, Dict[str, Any]]:
    """
    Prepares train/test splits, cleans features, and constructs the preprocessor.
    Optionally samples rows if specified for fast tuning while preserving class distribution.
    """
    target = detect_target_column(df, target_col)
    task_type = detect_task_type(df[target])
    numeric_cols, categorical_cols, dropped_cols = identify_columns(df, target)

    used_features = numeric_cols + categorical_cols
    X = df[used_features].copy()
    y = df[target].copy()

    # Drop rows where target itself is missing if any
    valid_idx = y.dropna().index
    X = X.loc[valid_idx]
    y = y.loc[valid_idx]

    # Optional sampling for rapid interactive training
    if sample_size and sample_size < len(X):
        if task_type == "classification":
            X, _, y, _ = train_test_split(
                X, y,
                train_size=sample_size,
                random_state=random_state,
                stratify=y
            )
        else:
            X, _, y, _ = train_test_split(
                X, y,
                train_size=sample_size,
                random_state=random_state
            )

    # Train / test split
    stratify = y if task_type == "classification" and y.nunique() <= 10 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify
    )

    preprocessor = build_preprocessor(numeric_cols, categorical_cols)

    metadata = {
        "target_column": target,
        "task_type": task_type,
        "numeric_features": numeric_cols,
        "categorical_features": categorical_cols,
        "dropped_features": dropped_cols,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "test_size": test_size,
        "random_state": random_state
    }

    return X_train, X_test, y_train, y_test, preprocessor, metadata
