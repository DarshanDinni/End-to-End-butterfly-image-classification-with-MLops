from butterflyClassifier.constants import *
from butterflyClassifier.entity.config_entity import *
from butterflyClassifier.utils.common import create_directories, read_yaml


class ConfigurationManager:
    def __init__(
        self,
        config_file_path=CONFIG_FILE_PATH,
        params_file_path=PARAMS_FILE_PATH,
        secret_file_path=SECRET_FILE_PATH,
    ):
        # Initialize the ConfigurationManager with paths to configuration, parameters, and secret files
        self.config = read_yaml(config_file_path)
        self.params = read_yaml(params_file_path)
        self.secrets = read_yaml(secret_file_path)

        # Create necessary directories based on the configuration
        create_directories([self.config.artifact_root])

    def get_params_config(self):
        # Get the parameter comfiguration from the main configuaration
        params = self.params

        # Create parameters Config object
        parameter_config = ParametersConfig(
            TEST_SIZE=params.TEST_SIZE, RANDOM_STATE=params.RANDOM_STATE
        )

        # Create necessary directories based on the configuration
        create_directories([self.config.data_preparation.root_path])

        return parameter_config

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        # Get the data ingestion configuration from the main configuration
        config = self.config.data_ingestion

        # Create directories specified in the data ingestion configuration
        create_directories([config.root_dir])

        # Create a DataIngestionConfig object with the relevant configuration parameters
        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_url=config.source_url,
            local_data_file=config.local_data_file,
        )

        return data_ingestion_config

    def get_data_preparation_config(self) -> DataPreparationConfig:
        # Get the config parameters for the data preparation
        config = self.config.data_preparation

        # Create DataPreparationConfig object with relevant configuration parameters
        get_data_preparation_config = DataPreparationConfig(
            root_path=config.root_path,
            train_csv_path=config.train_csv_path,
            test_csv_path=config.test_csv_path,
            train_images_path=config.train_images_path,
            test_images_path=config.test_csv_path,
            train_output_image_path=config.train_output_image_path,
            validation_output_image_path=config.validation_output_image_path,
            train_output_csv_path=config.train_output_csv_path,
            validation_output_csv_path=config.validation_output_csv_path,
        )

        return get_data_preparation_config

    def get_trainer_config(self) -> TrainerConfig:
        # Get the config and hyper parameters for the model training
        config = self.config.train
        params = self.params

        # Create directories for model
        create_directories([config.model_root_dir])

        # Create TrainerConfig object with relevant configuration parameters
        trainer_config = TrainerConfig(
            model_root_dir=config.model_root_dir,
            trained_model_path=config.trained_model_path,
            IMAGE_SIZE=params.IMAGE_SIZE,
            BATCH_SIZE=params.BATCH_SIZE,
            LEARNING_RATE=params.LEARNING_RATE,
            EPOCHS=params.EPOCHS,
        )

        return trainer_config
