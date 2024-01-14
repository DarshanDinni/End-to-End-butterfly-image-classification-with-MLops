import sys

from butterflyClassifier.components.data_preparation import DataPreparation
from butterflyClassifier.config.configuration import ConfigurationManager
from butterflyClassifier.exception_handling import CustomException
from butterflyClassifier.logger import logging

STAGE_NAME = "Data Preparation Stage"


class DataPreparationPipeline:
    def __init__(self) -> None:
        pass

    # Main method for the data preparation pipeline
    def main(self):
        try:
            # Create a ConfigurationManager to retrieve configuration details
            config = ConfigurationManager()

            # Get the data preparation configuration from the ConfigurationManager
            data_preparation_config = config.get_data_preparation_config()

            # Get the parameters configuration from the ConfigurationManager
            parameters_config = config.get_params_config()

            # Instantiate DataPreparation class
            data_preparation = DataPreparation(
                parameters_config, data_preparation_config
            )

            # Split the training dataset into train and validation set
            data_preparation.train_validation_set()

            # Organizing training data and validation data
            data_preparation.organize_data(
                data_preparation_config.train_output_csv_path,
                data_preparation_config.train_images_path,
                data_preparation_config.train_output_image_path,
            )
            data_preparation.organize_data(
                data_preparation_config.validation_output_csv_path,
                data_preparation_config.train_images_path,
                data_preparation_config.validation_output_image_path,
            )

        except Exception as e:
            logging.info(f"Error occured in {STAGE_NAME}")
            raise CustomException(e, sys)
