import logging
import os
import shutil
import sys

import pandas as pd
from sklearn.model_selection import train_test_split

from butterflyClassifier.entity.config_entity import *
from butterflyClassifier.exception_handling import CustomException


class DataPreparation:
    def __init__(
        self,
        params_config: ParametersConfig,
        data_preparation_config: DataPreparationConfig,
    ):
        self.params = params_config
        self.data_preparation = data_preparation_config

    def train_validation_set(self):
        try:
            # Load the csv file into a dataframe
            df = pd.read_csv(self.data_preparation.train_csv_path)

            # Split the dataset into training and validation dataset
            train_set_csv, validation_set_csv = train_test_split(
                df,
                test_size=self.params.TEST_SIZE,
                random_state=self.params.RANDOM_STATE,
                shuffle=True,
            )

            # Save the csv file to the specified location
            train_set_csv.to_csv(
                self.data_preparation.train_output_csv_path, index=False
            )
            validation_set_csv.to_csv(
                self.data_preparation.validation_output_csv_path, index=False
            )

            logging.info("Train and validation dataset csv completed")

        except Exception as e:
            raise CustomException(e, sys)

    def organize_data(self, csv_path, input_folder_path, output_folder_path):
        try:
            # Load CSV file into a DataFrame
            df = pd.read_csv(csv_path)

            # Create output folder if it doesn't exist
            os.makedirs(output_folder_path, exist_ok=True)

            # Iterate through each row in the DataFrame
            for index, row in df.iterrows():
                # Extract file name and label from the DataFrame
                file_name = row["filename"]
                label = row["label"]

                # Create subfolder for the label if it doesn't exist
                label_folder = os.path.join(output_folder_path, label)
                os.makedirs(label_folder, exist_ok=True)

                # Define paths for source and destination files
                source_path = os.path.join(input_folder_path, file_name)
                destination_path = os.path.join(label_folder, file_name)

                # Move the image file to the respective subfolder
                shutil.move(source_path, destination_path)

            logging.info(
                "Data organization for {0} completed".format(csv_path.split("/")[-1])
            )
        except Exception as e:
            raise CustomException(e, sys)
