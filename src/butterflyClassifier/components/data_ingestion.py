import os
import subprocess
import sys

import opendatasets as od

from butterflyClassifier.entity.config_entity import DataIngestionConfig
from butterflyClassifier.exception_handling import CustomException
from butterflyClassifier.logger import logging


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        # Initialize the DataIngestion object with a given configuration
        self.config = config

    def download_data(self):
        try:
            # Get dataset URL and local download directory from the configuration
            dataset_url = self.config.source_url
            zip_download_dir = self.config.local_data_file

            # Create the root directory if it doesn't exist
            os.makedirs(self.config.root_dir, exist_ok=True)

            # Log the start of the data download process
            logging.info(
                f"Data is getting downloaded from `{dataset_url}` to file `{zip_download_dir}`"
            )

            # Download the dataset using opendatasets and unzip it
            od.download(dataset_url, data_dir=zip_download_dir, unzip=True)

        except subprocess.CalledProcessError as e:
            # Handle any subprocess errors during the download process
            print(f"Error downloading Kaggle dataset: {e}")

        except Exception as e:
            # If any other exception occurs, raise a custom exception with details
            raise CustomException(e, sys)
