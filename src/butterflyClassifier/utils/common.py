import os
import sys
from pathlib import Path

import yaml
from box import ConfigBox
from box.exceptions import BoxValueError
from ensure import ensure_annotations

from butterflyClassifier.exception_handling import CustomException
from butterflyClassifier.logger import logging


@ensure_annotations
def read_yaml(yaml_file_path: Path) -> ConfigBox:
    """
    Read a YAML file and return a ConfigBox object.

    :param yaml_file_path: Path to the YAML file.
    :return: ConfigBox object representing the YAML data.
    """
    try:
        # Read YAML file and create a ConfigBox object
        with open(yaml_file_path, "r") as file:
            yaml_data = yaml.safe_load(file)
            logging.info(f"yaml file {yaml_file_path} loaded successfully")
            return ConfigBox(yaml_data)
    except BoxValueError:
        # Handle BoxValueError if the YAML file is empty
        raise CustomException("yaml file is empty", sys)
    except Exception as e:
        # Handle other exceptions and raise a custom exception
        raise CustomException(e, sys)


@ensure_annotations
def create_directories(path_to_directories: list, verbose=True):
    """
    Create a list of directories.

    :param path_to_directories: List of paths of directories.
    :param verbose: If True, log directory creation messages.
    """
    for path in path_to_directories:
        # Create directories and log messages if verbose is True
        os.makedirs(path, exist_ok=True)
        if verbose:
            logging.info(f"created directory at: {path}")
