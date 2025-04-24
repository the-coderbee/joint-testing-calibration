"""Model utilities for ML pipeline."""
import pandas as pd
import numpy as np
import joblib
import os


# Set up model path using absolute path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../..'))
MODEL_PATH = os.path.join(project_root, 'calibration_pipeline.joblib')
model = None


def load_model():
    """Load the trained model."""
    global model
    try:
        model = joblib.load(MODEL_PATH)
        print(f"Model loaded successfully from {MODEL_PATH}")
        return model
    except Exception as e:
        print(f"Error loading model from {MODEL_PATH}: {e}")
        raise


def get_model():
    """Get the trained model, loading it if needed."""
    global model
    if model is None:
        model = load_model()
    return model


def get_model_features():
    """Get model features for debugging."""
    model = get_model()
    try:
        # For scikit-learn pipeline with feature names
        if hasattr(model, 'feature_names_in_'):
            print("Model expects these features:")
            print(sorted(model.feature_names_in_.tolist()))
            return sorted(model.feature_names_in_.tolist())
        # For pipeline with a named step that has features
        elif hasattr(model, 'steps'):
            for name, step in model.steps:
                if hasattr(step, 'feature_names_in_'):
                    print(f"Step {name} expects these features:")
                    print(sorted(step.feature_names_in_.tolist()))
                    return sorted(step.feature_names_in_.tolist())
    except Exception as e:
        print(f"Couldn't extract model features: {e}")
    return []


def make_prediction(input_df):
    """Make prediction using the loaded model."""
    try:
        model = get_model()
        
        # Make prediction with model
        print(f"Available columns: {sorted(input_df.columns.tolist())}")
        print(f"Input dataframe shape: {input_df.shape}")
        
        # Check for NaN values
        nan_columns = input_df.columns[input_df.isna().any()].tolist()
        if nan_columns:
            print(f"Warning: NaN values found in columns: {nan_columns}")
            # Fill NaN values with 0 except for target variable
            for col in nan_columns:
                if col != 'true_strength':
                    input_df[col] = input_df[col].fillna(0)
        
        print(f"Final columns: {sorted(input_df.columns.tolist())}")
        
        # Get model features for debugging
        model_features = get_model_features()
        print(f"Model expects these features: {model_features}")
        
        # Check for missing columns required by the model
        if hasattr(model, 'feature_names_in_'):
            missing_cols = set(model.feature_names_in_) - set(input_df.columns)
            if missing_cols:
                print(f"Missing columns required by model: {missing_cols}")
                # You might want to try to add these columns with default values
                for col in missing_cols:
                    input_df[col] = 0
                    print(f"Added missing column '{col}' with default value 0")
        
        prediction = model.predict(input_df)
        print(f"Prediction successful: {prediction[0]}")
        return prediction[0]
    except Exception as e:
        import traceback
        print(f"Prediction error: {e}")
        print(traceback.format_exc())
        # If column error, check which columns are missing
        if "columns are missing" in str(e):
            missing_columns = eval(str(e).split("columns are missing: ")[1])
            print(f"Missing columns: {missing_columns}")
            print(f"Available columns: {sorted(input_df.columns.tolist())}")
        return None 