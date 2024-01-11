from butterflyClassifier.constants import *
from butterflyClassifier.entity.config_entity import DataIngestionConfig
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
