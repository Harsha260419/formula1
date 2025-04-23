import streamlit as st
import pandas as pd
import logging

# Import functions from your src modules
from src.utils.helpers import load_config
from src.data.data_ingestion import fetch_session_data
from src.data.data_preprocessing import preprocess_prediction_data # Need to update this
from src.model.prediction import load_model, predict_podium_probabilities, get_top_n_predictions, load_training_feature_names

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Streamlit App ---
st.set_page_config(layout="wide", page_title="Monaco GP 2025 Predictor")

st.title(" Monaco Grand Prix 2025 Top 3 Predictor")
st.markdown("""
Welcome to the 2025 Monaco GP Predictor! This tool uses a machine learning model
trained on historical race data to predict the drivers most likely to finish on the podium.

**Disclaimer:** Predictions are based on a simplified model and available session data (ideally Qualifying).
Real race outcomes are influenced by many unpredictable factors.
""")

# --- Configuration ---
config = load_config()
PREDICTION_YEAR = config['target_race']['year']
PREDICTION_EVENT = config['target_race']['event_name']
PREDICTION_SESSION = config['target_race']['prediction_session']
FASTF1_CACHE = config['fastf1_cache_dir']
MODEL_PATH = config['model_path']


# --- Helper functions for Streamlit caching ---

@st.cache_data
def cached_load_model(path):
     """Caches model loading."""
     return load_model(path)

@st.cache_data
def cached_load_feature_names(path):
    """Caches training feature names loading."""
    return load_training_feature_names(path)


@st.cache_data(ttl=600) # Cache for 10 minutes, adjust as needed
def cached_fetch_data(year, event, session, cache_dir):
     """Caches data fetching from FastF1."""
     logger.info(f"Attempting to fetch data for {year} {event} {session} with caching...")
     data = fetch_session_data(year, event, session, cache_dir)
     if data is None or data.empty:
          st.warning(f"Could not load data for {year} {event} {session}. Data might not be available yet.")
     return data

@st.cache_data
def cached_preprocess_data(data_df, training_features):
    """Caches data preprocessing."""
    if data_df is None or data_df.empty:
         return pd.DataFrame(), pd.DataFrame()
    logger.info("Attempting to preprocess data with caching...")
    return preprocess_prediction_data(data_df, training_features) # Pass training feature names


# --- Load Model ---
model = cached_load_model(MODEL_PATH)
training_feature_names = cached_load_feature_names(MODEL_PATH) # Load the list of expected features

if model is None or training_feature_names is None:
    st.error("Model or training features not loaded. Please ensure the model training script was run successfully.")
    st.stop() # Stop execution if model isn't available


# --- Get Data for Prediction ---
st.header("Get Data for Prediction")
st.write(f"Fetching data for: **{PREDICTION_YEAR} {PREDICTION_EVENT} - {PREDICTION_SESSION}**")

# Button to trigger data fetching and prediction
if st.button(f"Fetch {PREDICTION_SESSION} Data and Predict"):
    with st.spinner(f"Fetching and processing {PREDICTION_SESSION} data..."):
        # Fetch latest session data (ideally Qualifying)
        session_data_2025 = cached_fetch_data(
            PREDICTION_YEAR,
            PREDICTION_EVENT,
            PREDICTION_SESSION,
            FASTF1_CACHE
        )

        if session_data_2025 is not None and not session_data_2025.empty:
            st.success(f"Successfully fetched {PREDICTION_SESSION} data.")

            # Preprocess data for prediction
            # Pass the list of features the model expects
            prediction_features_2025, driver_map_2025 = cached_preprocess_data(
                session_data_2025,
                training_feature_names # Pass the list of features the model expects
            )

            if not prediction_features_2025.empty:
                st.success("Data preprocessed successfully.")

                # --- Make Predictions ---
                with st.spinner("Generating predictions..."):
                     predictions_df = predict_podium_probabilities(
                         model,
                         prediction_features_2025,
                         driver_map_2025
                     )

                     if not predictions_df.empty:
                        st.success("Predictions generated.")

                        # --- Display Results ---
                        st.header("Predicted Top 3 for Monaco Grand Prix 2025")

                        top_3_predictions = get_top_n_predictions(predictions_df, n=3)

                        if top_3_predictions:
                            col1, col2, col3 = st.columns(3)

                            # Find probabilities for all drivers to show table later
                            all_driver_predictions = predictions_df.sort_values(by='PodiumProbability', ascending=False).reset_index(drop=True)

                            with col1:
                                st.subheader("🥇 P1 Contender")
                                winner = top_3_predictions[0]
                                st.metric(label=winner['Driver'], value=f"{winner['PodiumProbability']:.1%}")
                                # Add team info if available in driver_map

                            if len(top_3_predictions) > 1:
                                with col2:
                                    st.subheader("🥈 P2 Contender")
                                    second = top_3_predictions[1]
                                    st.metric(label=second['Driver'], value=f"{second['PodiumProbability']:.1%}")

                            if len(top_3_predictions) > 2:
                                 with col3:
                                    st.subheader("🥉 P3 Contender")
                                    third = top_3_predictions[2]
                                    st.metric(label=third['Driver'], value=f"{third['PodiumProbability']:.1%}")

                            st.subheader("All Drivers' Podium Probabilities")
                            # Display the full sorted list
                            st.dataframe(all_driver_predictions, use_container_width=True, hide_index=True)


                        else:
                            st.warning("Could not determine top predictions.")

                     else:
                        st.error("Failed to generate predictions.")

            else:
                st.error("Failed to preprocess data for prediction. Check logs for details.")
        else:
            st.warning(f"Could not fetch data for {PREDICTION_YEAR} {PREDICTION_EVENT} {PREDICTION_SESSION}. Please ensure the session data is available via FastF1.")

st.markdown("---")
st.info("Model trained on historical Monaco GP data using a simplified approach. Accuracy is not guaranteed.")