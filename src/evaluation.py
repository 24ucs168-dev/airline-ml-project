"""
Model Evaluation Module
Computes performance metrics, confusion matrices, classification reports,
and feature importances for Random Forest models.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


def extract_feature_importances(pipeline: Pipeline, raw_feature_names: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Extracts feature importances from a trained Pipeline containing ColumnTransformer and RandomForest.
    Cleanly strips prefixes like 'num__' and 'cat__' for presentation.
    """
    try:
        model = pipeline.named_steps.get("model")
        preprocessor = pipeline.named_steps.get("preprocessor")

        if model is None or not hasattr(model, "feature_importances_"):
            return pd.DataFrame(columns=["feature", "importance"])

        importances = model.feature_importances_

        # Get feature names from preprocessor
        if preprocessor is not None and hasattr(preprocessor, "get_feature_names_out"):
            feature_names = preprocessor.get_feature_names_out()
            # Clean up prefixes
            cleaned_names = []
            for name in feature_names:
                clean = name
                for prefix in ["num__", "cat__", "remainder__"]:
                    if clean.startswith(prefix):
                        clean = clean[len(prefix):]
                cleaned_names.append(clean)
            feature_names = cleaned_names
        elif raw_feature_names and len(raw_feature_names) == len(importances):
            feature_names = raw_feature_names
        else:
            feature_names = [f"Feature {i+1}" for i in range(len(importances))]

        df_importance = pd.DataFrame({
            "feature": feature_names,
            "importance": importances
        }).sort_values(by="importance", ascending=False).reset_index(drop=True)

        return df_importance
    except Exception as e:
        print(f"Warning: Could not extract feature importances: {e}")
        return pd.DataFrame(columns=["feature", "importance"])


def evaluate_classification(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
    """
    Evaluates a classification model and returns a comprehensive metrics dictionary.
    """
    y_pred = pipeline.predict(X_test)
    classes = list(np.unique(np.concatenate([np.unique(y_test), np.unique(y_pred)])))

    acc = float(accuracy_score(y_test, y_pred))
    prec_weighted = float(precision_score(y_test, y_pred, average="weighted", zero_division=0))
    rec_weighted = float(recall_score(y_test, y_pred, average="weighted", zero_division=0))
    f1_weighted = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))

    prec_macro = float(precision_score(y_test, y_pred, average="macro", zero_division=0))
    rec_macro = float(recall_score(y_test, y_pred, average="macro", zero_division=0))
    f1_macro = float(f1_score(y_test, y_pred, average="macro", zero_division=0))

    cm = confusion_matrix(y_test, y_pred, labels=classes)
    report_dict = classification_report(y_test, y_pred, labels=classes, output_dict=True, zero_division=0)
    report_text = classification_report(y_test, y_pred, labels=classes, zero_division=0)

    # ROC AUC if binary and predict_proba is available
    roc_auc = None
    if len(classes) == 2 and hasattr(pipeline, "predict_proba"):
        try:
            pos_label = classes[1]
            y_proba = pipeline.predict_proba(X_test)[:, 1]
            roc_auc = float(roc_auc_score((y_test == pos_label).astype(int), y_proba))
        except Exception:
            roc_auc = None

    # Feature importances
    feat_imp_df = extract_feature_importances(pipeline, list(X_test.columns))

    return {
        "task_type": "classification",
        "classes": [str(c) for c in classes],
        "accuracy": round(acc, 4),
        "precision": round(prec_weighted, 4),
        "recall": round(rec_weighted, 4),
        "f1_score": round(f1_weighted, 4),
        "precision_macro": round(prec_macro, 4),
        "recall_macro": round(rec_macro, 4),
        "f1_macro": round(f1_macro, 4),
        "roc_auc": round(roc_auc, 4) if roc_auc is not None else None,
        "confusion_matrix": cm.tolist(),
        "classification_report_dict": report_dict,
        "classification_report_text": report_text,
        "feature_importances": feat_imp_df.to_dict(orient="records"),
        "top_features": feat_imp_df.head(15).to_dict(orient="records")
    }


def evaluate_regression(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
    """
    Evaluates a regression model and returns a comprehensive metrics dictionary.
    """
    y_pred = pipeline.predict(X_test)

    mae = float(mean_absolute_error(y_test, y_pred))
    mse = float(mean_squared_error(y_test, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_test, y_pred))

    feat_imp_df = extract_feature_importances(pipeline, list(X_test.columns))

    return {
        "task_type": "regression",
        "mae": round(mae, 4),
        "mse": round(mse, 4),
        "rmse": round(rmse, 4),
        "r2_score": round(r2, 4),
        "feature_importances": feat_imp_df.to_dict(orient="records"),
        "top_features": feat_imp_df.head(15).to_dict(orient="records")
    }


def evaluate_model(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    task_type: str = "classification"
) -> Dict[str, Any]:
    """
    Unified evaluation dispatch.
    """
    if task_type == "classification":
        return evaluate_classification(pipeline, X_test, y_test)
    else:
        return evaluate_regression(pipeline, X_test, y_test)
