import sys

from butterflyClassifier.exception_handling import CustomException
from butterflyClassifier.logger import logging
from butterflyClassifier.pipeline.data_ingestion_pipeline import \
    DataIngesionPipeline

# Define the stage name for logging
STAGE_NAME = "Data ingestion pipeline"

try:
    # Log the start of the data ingestion pipeline stage
    logging.info(f"Stage {STAGE_NAME} started")

    # Instantiate the DataIngesionPipeline class
    pipeline = DataIngesionPipeline()

    # Call the main method of the DataIngesionPipeline class to execute the data ingestion pipeline
    pipeline.main()

    # Log the completion of the data ingestion pipeline stage
    logging.info(f"Stage {STAGE_NAME} completed")

except Exception as e:
    # If an exception occurs during the execution, raise a CustomException with the exception details and sys information
    raise CustomException(e, sys)
