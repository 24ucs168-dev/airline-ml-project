"""
Model Training Module
Constructs, tunes, trains, and serializes the Random Forest Pipeline.
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from src.evaluation import evaluate_model


def get_default_hyperparameters(task_type: str = "classification") -> Dict[str, Any]:
    """
    Returns optimal default hyperparameters for Random Forest.
    """
    params = {
        "n_estimators": 100,
        "max_depth": 18,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "random_state": 42,
        "n_jobs": -1
    }
    if task_type == "classification":
        params["class_weight"] = "balanced_subsample"
    return params


def create_random_forest_model(task_type: str = "classification", **params):
    """
    Instantiates RandomForestClassifier or RandomForestRegressor with specified hyperparameters.
    """
    if task_type == "classification":
        return RandomForestClassifier(**params)
    else:
        # Filter out classification-only params if any
        reg_params = {k: v for k, v in params.items() if k != "class_weight"}
        return RandomForestRegressor(**reg_params)


def build_pipeline(preprocessor: ColumnTransformer, model) -> Pipeline:
    """
    Wraps the column transformer and model into an end-to-end sklearn Pipeline.
    """
    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])


def train_pipeline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    preprocessor: ColumnTransformer,
    task_type: str = "classification",
    hyperparameters: Optional[Dict[str, Any]] = None
) -> Tuple[Pipeline, float]:
    """
    Fits the end-to-end Pipeline and measures execution time.
    """
    params = get_default_hyperparameters(task_type)
    if hyperparameters:
        params.update(hyperparameters)

    model = create_random_forest_model(task_type=task_type, **params)
    pipeline = build_pipeline(preprocessor, model)

    start_time = time.time()
    pipeline.fit(X_train, y_train)
    duration = time.time() - start_time

    return pipeline, duration


def save_model_artifacts(
    pipeline: Pipeline,
    metadata: Dict[str, Any],
    evaluation_results: Optional[Dict[str, Any]] = None,
    models_dir: str = "models"
) -> Tuple[str, str]:
    """
    Saves the trained pipeline as .pkl and full metadata & metrics as .json.
    """
    out_dir = Path(models_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = out_dir / "random_forest_model.pkl"
    meta_path = out_dir / "model_metadata.json"

    # Serialize pipeline
    joblib.dump(pipeline, str(model_path))

    # Combine metadata
    full_meta = {
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_type": "RandomForestClassifier" if metadata.get("task_type") == "classification" else "RandomForestRegressor",
        **metadata
    }
    if evaluation_results:
        full_meta["evaluation"] = evaluation_results

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(full_meta, f, indent=2)

    return str(model_path.resolve()), str(meta_path.resolve())


def load_model_artifacts(models_dir: str = "models") -> Optional[Tuple[Pipeline, Dict[str, Any]]]:
    """
    Loads saved model pipeline and metadata if available.
    """
    out_dir = Path(models_dir)
    model_path = out_dir / "random_forest_model.pkl"
    meta_path = out_dir / "model_metadata.json"

    if not model_path.exists():
        return None

    try:
        pipeline = joblib.load(str(model_path))
        metadata = {}
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        return pipeline, metadata
    except Exception as e:
        print(f"Error loading model artifacts: {e}")
        return None


def run_full_training(
    df: pd.DataFrame,
    target_col: Optional[str] = None,
    test_size: float = 0.2,
    sample_size: Optional[int] = None,
    hyperparameters: Optional[Dict[str, Any]] = None,
    models_dir: str = "models"
) -> Dict[str, Any]:
    """
    End-to-end training orchestrator:
    Prepares data -> Fits Pipeline -> Evaluates on Test Set -> Saves Artifacts.
    """
    from src.data_preprocessing import prepare_data

    random_state = 42
    if hyperparameters and "random_state" in hyperparameters:
        random_state = hyperparameters["random_state"]

    X_train, X_test, y_train, y_test, preprocessor, meta = prepare_data(
        df=df,
        target_col=target_col,
        test_size=test_size,
        random_state=random_state,
        sample_size=sample_size
    )

    pipeline, train_duration = train_pipeline(
        X_train=X_train,
        y_train=y_train,
        preprocessor=preprocessor,
        task_type=meta["task_type"],
        hyperparameters=hyperparameters
    )

    eval_results = evaluate_model(
        pipeline=pipeline,
        X_test=X_test,
        y_test=y_test,
        task_type=meta["task_type"]
    )

    meta["training_duration_seconds"] = round(train_duration, 3)
    meta["hyperparameters"] = hyperparameters or get_default_hyperparameters(meta["task_type"])

    model_path, meta_path = save_model_artifacts(
        pipeline=pipeline,
        metadata=meta,
        evaluation_results=eval_results,
        models_dir=models_dir
    )

    return {
        "pipeline": pipeline,
        "metadata": meta,
        "evaluation": eval_results,
        "model_path": model_path,
        "meta_path": meta_path,
        "X_test": X_test,
        "y_test": y_test
    }
