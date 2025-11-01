# config/__init__.py
import logging
import sys
import os
from datetime import datetime

# --- Basic Configuration ---
LOG_LEVEL = logging.INFO
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# --- Determine Log Directory ---
# Use a platform-independent approach to find a suitable log directory.
log_directory = os.path.join(
    os.path.expanduser("~"), "logs", "cognitive_language_model_logs"
)
os.makedirs(log_directory, exist_ok=True)

# --- Create a unique log file for each run ---
log_filename = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
log_filepath = os.path.join(log_directory, log_filename)

# --- Get the Root Logger ---
# Configuring the root logger ensures that all loggers in your application
# will inherit this configuration.
logger = logging.getLogger()
logger.setLevel(LOG_LEVEL)

# --- Create Handlers ---
# 1. A handler to write log messages to a file
file_handler = logging.FileHandler(log_filepath)
file_handler.setFormatter(logging.Formatter(LOG_FORMAT))

# 2. A handler to stream log messages to the console (stdout)
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(logging.Formatter(LOG_FORMAT))

# --- Add Handlers to the Logger ---
# Avoid adding handlers if they already exist (e.g., in interactive sessions)
if not logger.handlers:
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

# --- Initial Log Message ---
logger.info(f"Logging initialized. Log file: {log_filepath}")