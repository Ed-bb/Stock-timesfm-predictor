"""Configuration for TimesFM Stock Predictor"""
import os
from dotenv import load_dotenv

load_dotenv()

# Default stock ticker
DEFAULT_TICKER = "BWET"

# TimesFM Configuration
TIMESFM_CONFIG = {
    "checkpoint_path": "google/timesfm-3.0-pytorch",
    "per_core_batch_size": 32,
    "device": os.getenv("TIMESFM_DEVICE", "cuda"),
}

# Forecasting Parameters
FORECAST_HORIZON = int(os.getenv("FORECAST_HORIZON", "30"))
HISTORICAL_PERIOD = os.getenv("HISTORICAL_PERIOD", "2y")
MIN_DATA_POINTS = 60

# Visualization
PLOT_DPI = 150
PLOT_FIGSIZE = (14, 6)

# Paths
DATA_DIR = "data"
LOGS_DIR = "logs"
PLOTS_DIR = "plots"

for directory in [DATA_DIR, LOGS_DIR, PLOTS_DIR]:
    os.makedirs(directory, exist_ok=True)
