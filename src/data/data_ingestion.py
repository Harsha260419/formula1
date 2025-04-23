import fastf1 as ff1
from fastf1 import plotting
import pandas as pd
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Enable cache (configure directory via config)
# ff1.Cache.enable_cache('./cache') # Use config value in actual implementation

def fetch_session_data(year, event_name, session_name, cache_dir):
    """Fetches data for a specific session."""
    ff1.Cache.enable_cache(cache_dir)
    logger.info(f"Fetching {session_name} data for {year} {event_name}...")
    try:
        session = ff1.get_session(year, event_name, session_name)
        session.load() # Load all available session data
        logger.info("Data loaded successfully.")

        # Example: Get lap data - you might need other data like telemetry, session status etc.
        # Depending on the session, lap data structure varies.
        # For Qualifying, maybe get the best lap for each driver.
        if session_name == 'Qualifying':
             laps = session.laps.pick_quicklaps(n=1).reset_index(drop=True)
             # Add driver number and team for easier joining later
             laps['Driver'] = laps['Driver']
             laps['Team'] = laps['Team']
             return laps[['Driver', 'Team', 'LapTime', 'Sector1Time', 'Sector2Time', 'Sector3Time', 'SpeedI1', 'SpeedI2', 'SpeedFL', 'SpeedST']] # Example features
        elif session_name == 'Race':
             laps = session.laps.pick_wo_ignored().reset_index(drop=True)
             return laps # Need to aggregate race data for training
        else: # Practice sessions
             laps = session.laps.pick_wo_ignored().reset_index(drop=True)
             return laps # Need to aggregate practice data


    except Exception as e:
        logger.error(f"Error fetching data for {year} {event_name} {session_name}: {e}")
        # Handle cases where data is not available (e.g., 2025 data not loaded yet)
        return None

def fetch_historical_race_data(years, event_name, cache_dir):
    """Fetches race data for multiple years for training."""
    all_race_data = []
    for year in years:
        logger.info(f"Fetching Race data for {year} {event_name}...")
        race_data = fetch_session_data(year, event_name, 'Race', cache_dir)
        if race_data is not None:
            race_data['Year'] = year # Add year for context
            all_race_data.append(race_data)

    if not all_race_data:
        logger.warning("No historical race data fetched.")
        return pd.DataFrame()

    return pd.concat(all_race_data, ignore_index=True)

# Example usage (for training script)
# if __name__ == "__main__":
#    config = load_config()
#    historical_data = fetch_historical_race_data(config['training_data_years'], config['target_race']['event_name'], config['fastf1_cache_dir'])
#    print(historical_data.head())