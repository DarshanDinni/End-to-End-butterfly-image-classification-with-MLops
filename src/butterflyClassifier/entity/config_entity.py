from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    """
    Data class representing the configuration for data ingestion.

    Attributes:
        root_dir (Path): The root directory where data will be stored.
        source_url (str): The URL from which the data will be downloaded.
        local_data_file (Path): The local path where the downloaded data will be stored.
    """

    root_dir: Path
    source_url: str
    local_data_file: Path


@dataclass(frozen=True)
class DataPreparationConfig:
    """
    Data class representing the configuration for data preparation

    Attributes:
        root_path (Path): The root directory where all the data regarding data preparation will be stored.
        train_csv_path (Path): Path for training csv information is stored.
        test_csv_path (Path): Path for testing csv information is stored.
        train_images_path (Path): Path for where all the training images are stored.
        test_images_path (Path): Path for where all the testing images are stored.
        train_output_image_path (Path): Path for where the training images will be stored after data manipulation.
        validation_output_image_path (Path): Path for where the validation images will be stored after data manipulation.
        train_output_csv_path (Path): Path to save the training csv after data manipulation.
        validation_output_csv_path (Path): Path to save the validation csv after data manipulation.
    """

    root_path: Path
    train_csv_path: Path
    test_csv_path: Path
    train_images_path: Path
    test_images_path: Path
    train_output_image_path: Path
    validation_output_image_path: Path
    train_output_csv_path: Path
    validation_output_csv_path: Path


@dataclass(frozen=True)
class ParametersConfig:
    """
    Data class representing the configuration for parameters

    Attributes:
        TEST_SIZE (float): floating point number to represent the testing size for the train test split operation.
        RANDOM_STATE (int): integer number to create a random seed.
    """

    TEST_SIZE: float
    RANDOM_STATE: int


@dataclass(frozen=True)
class TrainerConfig:
    model_root_dir: Path
    trained_model_path: Path
    IMAGE_SIZE: int
    BATCH_SIZE: int
    LEARNING_RATE: float
    EPOCHS: int
