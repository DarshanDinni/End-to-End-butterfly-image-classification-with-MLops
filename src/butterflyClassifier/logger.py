import logging
import os
from datetime import datetime


def logger():
    # Specify the directory for log files
    log_dir = "logs"

    # Create a timestamp for the log file name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{timestamp}.log"

    # Specify the path for the log file
    log_path = os.path.join(log_dir, log_filename)

    # Create the logs directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)

    # Define the log message format
    log_format = "[%(asctime)s: %(levelname)s: %(module)s: %(message)s]"

    if not logging.root.handlers:
        # Configure the logging settings
        logging.basicConfig(
            level=logging.INFO,  # Set the logging level to INFO
            format=log_format,
            filename=log_path,  # Specify the path for the log file
        )
