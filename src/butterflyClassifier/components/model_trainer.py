import logging
import sys
from pathlib import Path
from timeit import default_timer as timer
from typing import List

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from tqdm.auto import tqdm

from butterflyClassifier.entity.config_entity import TrainerConfig
from butterflyClassifier.exception_handling import CustomException


class ModelTrainer:
    def __init__(self, trainer_config: TrainerConfig):
        self.model_trainer_config = trainer_config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def create_transform(self):
        transform = transforms.Compose(
            [
                transforms.Resize(self.model_trainer_config.IMAGE_SIZE),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        return transform

    def create_data_loader(
        self,
        train_data_path: Path,
        validation_data_path: Path,
        transform: transforms.Compose,
        batch_size: int,
        num_workers: int,
    ) -> tuple:
        """
        Create data loaders for training and validation datasets.

        Parameters:
            train_data_path (Path): Path to the training dataset.
            validation_data_path (Path): Path to the validation dataset.
            transform (transforms.Compose): Data transformation to be applied on the images.

        Returns:
            tuple: A tuple containing training DataLoader, validation DataLoader, and list of class names.
        """
        try:
            # Creating ImageFolder datasets for training and validation
            train_dataset = datasets.ImageFolder(
                root=train_data_path, transform=transform, target_transform=None
            )
            validation_dataset = datasets.ImageFolder(
                root=validation_data_path, transform=transform
            )

            # Creating DataLoaders
            train_dataloader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
            )
            validation_dataloader = DataLoader(
                validation_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
            )

            # Extracting class names from the training dataset
            class_names = train_dataset.classes

            logging.info("Created data loader for train and validation datasets")

            return train_dataloader, validation_dataloader, class_names
        except Exception as e:
            raise CustomException(e, sys)

    def train_and_test_loop(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        loss_function: nn.Module,
        optimizer: torch.optim.Optimizer,
        accuracy_fn: callable,
        device: torch.device,
        mode: str,
    ) -> tuple:
        """
        Train or evaluate the model based on the provided mode.

        Parameters:
        - model (nn.Module): The neural network model to be trained or evaluated.
        - dataloader (DataLoader): DataLoader providing the data.
        - loss_function (nn.Module): Loss function for optimization or evaluation.
        - optimizer (torch.optim.Optimizer): The optimizer for updating model parameters (only used in training mode).
        - accuracy_fn (callable): A function to calculate accuracy.
        - device (torch.device): Device (CPU or GPU) to perform training or evaluation.
        - mode (str): Either 'train' or 'test' to indicate the current mode.

        Returns:
        - tuple: A tuple containing the average loss and accuracy over all batches.
        """

        try:
            if mode == "train":
                # Set the model for training phase
                model.train()
            elif mode == "test":
                # Set the model for evaluation/testing phase
                model.eval()
            else:
                raise ValueError("Invalid mode. Use 'train' or 'test'.")

            # Initialize variables to store cumulative loss and accuracy
            total_loss, total_accuracy = 0, 0
            # Iterate over batches in the dataloader
            for X, y in dataloader:
                # Set the data to the specified device
                X, y = X.to(device), y.to(device)
                # Perform the forward pass
                prediction = model(X)
                # Calculate and accumulate the loss and accuracy over each batch
                loss = loss_function(prediction, y)
                total_loss += loss.item()
                total_accuracy += accuracy_fn(
                    y, torch.argmax(torch.softmax(prediction, dim=1), dim=1)
                )
                if mode == "train":
                    # Set the optimizer gradients to zero at the start of each batch training
                    optimizer.zero_grad()

                    # Perform backpropagation
                    loss.backward()

                    # Take a step towards minimum loss
                    optimizer.step()
            # Calculate the average loss and accuracy over all batches
            average_loss = total_loss / len(dataloader)
            average_accuracy = total_accuracy / len(dataloader)

            return average_loss, average_accuracy
        except Exception as e:
            raise CustomException(e, sys)

    def train_model(
        self,
        train_dataloader: DataLoader,
        validation_dataloader: DataLoader,
        class_names: List[str],
        accuracy_fn: callable,
    ):
        """
        Train the model using transfer learning with a pre-trained ResNet50 backbone.

        Parameters:
        - train_dataloader: DataLoader providing the training data.
        - validation_dataloader: DataLoader providing the validation data.
        - class_names: List of class names corresponding to the model's output.
        - accuracy_fn: A function to calculate accuracy.

        Returns:
        - dict: A dictionary containing training and testing results.
        - model: Model trained
        """

        try:
            result = {
                "train_loss": [],
                "train_accuracy": [],
                "test_loss": [],
                "test_accuracy": [],
            }

            # Load pre-trained ResNet50 model with weights from the default configuration
            model_resnet50_weights = models.ResNet50_Weights.DEFAULT
            model_resnet50 = models.resnet50(weights=model_resnet50_weights).to(
                self.device
            )

            # Freeze parameters to avoid unnecessary computations during training
            for param in model_resnet50.parameters():
                param.requires_grad = False

            # Modify the fully connected layer of resnet50 model to match the number of classes
            model_resnet50.fc = nn.Linear(
                in_features=2048, out_features=len(class_names)
            ).to(self.device)

            # Define the loss function and optimizer for the model
            loss_function = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(
                model_resnet50.parameters(), lr=self.model_trainer_config.LEARNING_RATE
            )

            start_time = timer()

            for epoch in tqdm(range(self.model_trainer_config.EPOCHS)):
                # Training phase
                train_results = self.train_and_test_loop(
                    model=model_resnet50,
                    dataloader=train_dataloader,
                    loss_function=loss_function,
                    optimizer=optimizer,
                    accuracy_fn=accuracy_fn,
                    device=self.device,
                    mode="train",
                )
                # Testing phase
                test_results = self.train_and_test_loop(
                    model=model_resnet50,
                    dataloader=validation_dataloader,
                    loss_function=loss_function,
                    optimizer=optimizer,
                    accuracy_fn=accuracy_fn,
                    device=self.device,
                    mode="test",
                )

                # Store results
                result["train_loss"].append(train_results[0])
                result["train_accuracy"].append(train_results[1])
                result["test_loss"].append(test_results[0])
                result["test_accuracy"].append(test_results[1])

                print(f"Epoch: {epoch}")
                print(
                    f"Training Loss: {train_results[0]}, Training Accuracy: {train_results[1]}"
                )
                print(
                    f"Testing Loss: {test_results[0]}, Testing Accuracy: {test_results[1]}"
                )

            end_timer = timer()

            print(f"Time taken to complete the training: {end_timer-start_time}")

            logging.info("Model trained and tested on the data.")
            return result, model_resnet50
        except Exception as e:
            raise CustomException(e, sys)

    def save_model(self, model: nn.Module, save_path: Path):
        """
        Function to save the model

        Parameters:
        - model: Model to save
        - save_path: Path to save the model
        """
        try:
            torch.save(model.state_dict(), save_path)
            logging.info("Model saved")
        except Exception as e:
            raise CustomException(e, sys)
