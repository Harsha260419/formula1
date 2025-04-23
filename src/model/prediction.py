import joblib
import pandas as pd
import logging
import os

logger = logging.getLogger(__name__)

def load_model(model_path):
    """Loads the trained model from disk."""
    if not os.path.exists(model_path):
        logger.error(f"Model file not found at {model_path}")
        return None
    logger.info(f"Loading model from {model_path}")
    try:
        model = joblib.load(model_path)
        logger.info("Model loaded successfully.")
        return model
    except Exception as e:
        logger.error(f"Error loading model from {model_path}: {e}")
        return None

def load_training_feature_names(model_path):
    """Loads the list of feature names the model was trained on."""
    feature_list_path = model_path.replace('.pkl', '_features.pkl')
    if not os.path.exists(feature_list_path):
         logger.error(f"Training feature list file not found at {feature_list_path}")
         return None
    try:
        feature_names = joblib.load(feature_list_path)
        logger.info("Training feature names loaded.")
        return feature_names
    except Exception as e:
        logger.error(f"Error loading training feature names from {feature_list_path}: {e}")
        return None


def predict_podium_probabilities(model, preprocessed_prediction_features, driver_mapping):
    """
    Makes predictions using the loaded model.
    Returns a DataFrame with Driver and their predicted podium probability.
    """
    if model is None or preprocessed_prediction_features.empty or driver_mapping.empty:
        logger.warning("Missing model or data for prediction.")
        return pd.DataFrame()

    logger.info(f"Making predictions for {preprocessed_prediction_features.shape[0]} drivers...")

    # Ensure columns match the order and names the model expects
    # This was handled (simplified) in preprocess_prediction_data,
    # but it's critical in a real scenario.
    # You would load the training feature names list here and reindex/reorder
    # preprocessed_prediction_features accordingly.
    # Example: preprocessed_prediction_features = preprocessed_prediction_features[training_feature_names]

    try:
        # Predict probabilities ([:, 1] gets the probability of the positive class, which is 1: IsOnPodium)
        probabilities = model.predict_proba(preprocessed_prediction_features)[:, 1]

        # Combine results with driver names
        predictions_df = driver_mapping.copy()
        predictions_df['PodiumProbability'] = probabilities

        logger.info("Predictions generated successfully.")
        return predictions_df

    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        # Log details about data shape and column mismatch if possible
        logger.error(f"Prediction data shape: {preprocessed_prediction_features.shape}")
        # logger.error(f"Prediction data columns: {preprocessed_prediction_features.columns.tolist()}") # Can be useful for debugging
        return pd.DataFrame()


def get_top_n_predictions(predictions_df, n=3):
    """Sorts predictions and returns the top N drivers."""
    if predictions_df.empty:
        logger.warning("No predictions available to rank.")
        return []

    # Sort by probability in descending order
    top_n_predictions = predictions_df.sort_values(by='PodiumProbability', ascending=False).head(n)

    logger.info(f"Top {n} predictions determined.")
    return top_n_predictions.to_dict('records') # Return as list of dicts