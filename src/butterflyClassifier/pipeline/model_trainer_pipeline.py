import os
import sys

from butterflyClassifier.components.model_trainer import ModelTrainer
from butterflyClassifier.config.configuration import ConfigurationManager
from butterflyClassifier.exception_handling import CustomException
from butterflyClassifier.utils.common import accuracy_fn


class ModelTrainerPipeline:
    def __init__(self) -> None:
        pass

    def main(self):
        try:
            # Initialize configuration manager
            config = ConfigurationManager()

            # Fetch model trainer configurations
            data_preparation_config = config.get_data_preparation_config()
            model_trainer_config = config.get_trainer_config()

            # Initialize ModelTrainer instance with the model trainer configuration
            model_trainer = ModelTrainer(model_trainer_config)

            # Create data transformation
            transform = model_trainer.create_transform()

            # Create data loaders for training and validation datasets
            (
                train_dataloader,
                validation_dataloader,
                class_names,
            ) = model_trainer.create_data_loader(
                data_preparation_config.train_output_image_path,
                data_preparation_config.validation_output_image_path,
                transform,
                model_trainer_config.BATCH_SIZE,
                os.cpu_count(),
            )

            # Train the model and get results and trained model
            result, trained_model = model_trainer.train_model(
                train_dataloader, validation_dataloader, class_names, accuracy_fn
            )

            # Save the trained model
            model_trainer.save_model(
                trained_model, model_trainer_config.trained_model_path
            )
        except Exception as e:
            raise CustomException(e, sys)
