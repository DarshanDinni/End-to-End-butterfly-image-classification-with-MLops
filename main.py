import logging
import sys

from butterflyClassifier.exception_handling import CustomException
from butterflyClassifier.logger import logger
from butterflyClassifier.pipeline.data_ingestion_pipeline import \
    DataIngesionPipeline
from butterflyClassifier.pipeline.data_preparation_pipeline import \
    DataPreparationPipeline
from butterflyClassifier.pipeline.model_trainer_pipeline import \
    ModelTrainerPipeline

if __name__ == "__main__":
    logger()

    # Define the stage name for logging
    STAGE_1 = "Data ingestion pipeline"

    try:
        # Log the start of the data ingestion pipeline stage
        logging.info(f"Stage {STAGE_1} started")

        # Instantiate the DataIngesionPipeline class
        ingestion_pipeline = DataIngesionPipeline()

        # Call the main method of the DataIngesionPipeline class to execute the data ingestion
        ingestion_pipeline.main()

        # Log the completion of the data ingestion pipeline stage
        logging.info(f"Stage {STAGE_1} completed")

    except Exception as e:
        # If an exception occurs during the execution, raise a CustomException with the exception details and sys information
        raise CustomException(e, sys)

    STAGE_2 = "Data preparation pipeline"

    try:
        # Log the start of the DataPreparationPipeline stage
        logging.info(f"Stage {STAGE_2} started")

        # Instantiate the DataPreparationPipeline class
        preparation_pipeline = DataPreparationPipeline()

        # Call the main method of the DataPreparationPipeline class to execute the data preparation
        preparation_pipeline.main()

        # Log the completion of the data ingestion pipeline stage
        logging.info(f"Stage {STAGE_2} completed")

    except Exception as e:
        # If an exception occurs during the execution, raise a CustomException with the exception details and sys information
        raise CustomException(e, sys)

    STAGE_3 = "Model training pipeline"

    try:
        # Log the start of the ModelTrainerPipeline stage
        logging.info(f"{STAGE_3} started")

        # Instantiate the ModelTrainerPipeline class
        model_trainer_pipeline = ModelTrainerPipeline()

        # Start the model training and testing
        model_trainer_pipeline.main()

        # Log the end of the ModelTrainerPipeline stage
        logging.info(f"{STAGE_3} completed")

    except Exception as e:
        raise CustomException(e, sys)
