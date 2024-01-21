import logging

from butterflyClassifier.components.data_ingestion import DataIngestion
from butterflyClassifier.config.configuration import ConfigurationManager
from butterflyClassifier.exception_handling import CustomException

STAGE_NAME = "Data Ingestion Stage"


class DataIngesionPipeline:
    def __init__(self) -> None:
        pass

    def main(self):
        # Main method for the data ingestion pipeline
        try:
            # Create a ConfigurationManager to retrieve configuration details
            config = ConfigurationManager()

            # Get the data ingestion configuration from the ConfigurationManager
            data_ingestion_config = config.get_data_ingestion_config()

            # Instantiate DataIngestion class with the data ingestion configuration
            data_ingestion = DataIngestion(data_ingestion_config)

            # Execute the data download process
            data_ingestion.download_data()

        except CustomException as e:
            # Handle custom exceptions and log the error message
            logging.error(f"Error in {STAGE_NAME}: {e}")

        except Exception as e:
            # Catch any other exceptions and log the error message
            logging.error(f"Unexpected error in {STAGE_NAME}: {e}")
