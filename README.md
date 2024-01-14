# Butterfly Image Classifier

## Overview

The Butterfly Image Classification project revolves around a diverse dataset featuring 75 distinct classes of butterflies. Comprising over 1000 labeled images, including validation samples, each image is exclusively associated with a single butterfly category.

The training dataset, encapsulated in Training_set.csv, contains labeled images crucial for model training. Additionally, the Testing_set.csv includes a list of images in the test folder, where the objective is to predict the corresponding labels.

This project aims to develop a robust machine learning model capable of accurately classifying butterfly species based on images.

## Table of Contents
- [Title and Overview](#butterfly-image-classifier)
- [Table of Contents](#table-of-contents)
- [Installation](#installation)
- [Components](#components)
- [Usage](#usage)
- [Contribution](#contribution)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Installation
1. **Clone the repository:**

    ```
    https://github.com/DarshanDinni/End-to-End-butterfly-image-classification-with-MLops.git
    ```

2. **Install dependencies:** 

    ```
    pip3 install -r requirements.txt --index-url https://download.pytorch.org/whl/cu118
    ```
3. **Run the project:**
    ```
    python main.py
    ```
    
    **Note:** Upon executing the command, a prompt will appear, requesting your Kaggle credentials. For guidance on obtaining these credentials, please refer to this [link](https://github.com/Kaggle/kaggle-api#api-credentials).

## Workflow

The following steps outline the process of creating each component within the End-to-End project, providing essential information for a comprehensive understanding.

- Update the config.yaml
- Update the secrets.yaml (optional)
- Update the params.yaml
- Create an entity 
- Update the configuration manager in src/butterflyClassifer
- Update the component
- Update the pipeline
- Update the main.py
- Update the dvc.yaml

## Components
### 1. Data Ingestion
The Data Ingestion component is tasked with efficiently downloading the dataset for model training. I have adhered to the aforementioned workflow to develop modular code, ultimately executing the pipeline to initiate the data download process.

### 2. Data Preparation
This section provides guidelines on how to prepare and organize the data for use in this project. Proper data organization ensures a smooth workflow during model training and evaluation.

**Instantiate DataPreparation Class:**
Create an instance of the `DataPreparation` class using the obtained configurations.

**Split Training Dataset:**
Use the `train_validation_set()` method to split the training csv dataset into training and validation sets.

**Organize Training and Validation Data:**
The `organize_data()` function is responsible for structuring and arranging the training and validation data. It creates organized folders based on provided output paths, segregating images into labeled subfolders. This step ensures a well-organized dataset, a prerequisite for effective machine learning model training and evaluation.

## Usage

## Contribution

## License
This project is licensed under the MIT License.

## Acknowledgements
- **PyTorch:** [Pytorch docs](https://pytorch.org/docs/stable/index.html)
- **Data source:** [Kaggle](https://www.kaggle.com/) 
- **Dataset link:** [Click here](https://www.kaggle.com/datasets/phucthaiv02/butterfly-image-classification/data)