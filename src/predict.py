"""
Prediction Module
Provides inference services, class probability extraction, and preset sample generation.
"""

from typing import Dict, Any, Union, Optional
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline


def predict_sample(
    pipeline: Pipeline,
    input_data: Union[Dict[str, Any], pd.DataFrame],
    task_type: str = "classification"
) -> Dict[str, Any]:
    """
    Runs prediction on a single record or multiple records using the trained pipeline.
    Returns prediction, probabilities, confidence score, and formatted details.
    """
    if isinstance(input_data, dict):
        df_input = pd.DataFrame([input_data])
    else:
        df_input = input_data.copy()

    # Predict class or value
    pred_raw = pipeline.predict(df_input)
    prediction = pred_raw[0]

    result = {
        "prediction": prediction if not isinstance(prediction, (np.generic,)) else prediction.item(),
        "task_type": task_type,
        "is_classification": (task_type == "classification")
    }

    # Extract probabilities if classification and pipeline supports predict_proba
    if task_type == "classification" and hasattr(pipeline, "predict_proba"):
        try:
            probas = pipeline.predict_proba(df_input)[0]
            classes = pipeline.classes_
            prob_dict = {
                str(cls_name): round(float(prob), 4)
                for cls_name, prob in zip(classes, probas)
            }
            max_prob = float(np.max(probas))
            
            result["probabilities"] = prob_dict
            result["confidence_score"] = round(max_prob * 100, 2)
            result["confidence_ratio"] = max_prob
        except Exception as e:
            result["probabilities"] = None
            result["confidence_score"] = None
            result["error_proba"] = str(e)

    return result


def get_default_feature_values(df: pd.DataFrame, feature_cols: list) -> Dict[str, Any]:
    """
    Computes sensible default values (median for numeric, mode for categorical)
    for each feature from the training dataset.
    """
    defaults = {}
    for col in feature_cols:
        if col not in df.columns:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            defaults[col] = float(df[col].median())
        else:
            mode_val = df[col].mode()
            defaults[col] = mode_val.iloc[0] if not mode_val.empty else ""
    return defaults


def get_preset_samples(df: pd.DataFrame, target_col: str) -> Dict[str, Dict[str, Any]]:
    """
    Finds exemplar rows from the dataset (e.g., highly satisfied business passenger vs dissatisfied passenger)
    to allow 1-click test scenarios in the UI.
    """
    presets = {}
    try:
        # Sample for each unique target class
        for label in df[target_col].dropna().unique()[:3]:
            subset = df[df[target_col] == label]
            if not subset.empty:
                row = subset.drop(columns=[target_col], errors="ignore").iloc[0].to_dict()
                # Clean numpy types
                clean_row = {}
                for k, v in row.items():
                    if pd.isna(v):
                        clean_row[k] = 0
                    elif isinstance(v, (np.integer, int)):
                        clean_row[k] = int(v)
                    elif isinstance(v, (np.floating, float)):
                        clean_row[k] = float(v)
                    else:
                        clean_row[k] = str(v)
                presets[f"Realistic Sample ({label})"] = clean_row
    except Exception as e:
        print(f"Warning: could not extract preset samples: {e}")

    return presets
