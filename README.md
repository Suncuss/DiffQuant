This is a python program that quantifies myogenic differentiation of C2C12 cells using phase contrast images.

## Introduction

DiffQuant is a machine learning-based tool designed to quantify C2C12 cell differentiation by predicting continuous values from phase contrast microscope images. This  approach offers an alternative method for measuring differentiation progress without relying on fluorescence microscopy. Developed using PyTorch and based on the ResNet-18 model, DiffQuant can be readily adapted to address other image classification or regression tasks.

## Table of Contents

- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Image Preprocessing](#image-preprocessing)
- [Training](#training)
- [Prediction](#prediction)
- [Contributing](#contributing)
- [License](#license)

## Installation

To install and run the project, follow these steps:

1. Clone the GitHub repository:

    ```
    git clone https://github.com/Suncuss/DiffQuant.git
    ```

2. To build and run docker image:

    ```
    sh docker/build_docker_img.sh
    sh docker/run_docker.sh
    ```


3. Run the project:

    ```
    python main.py
    ```


## Data Preparation

The images need to be preprocessed for both training and prediction. Specifically, each image file will be cropped into five square images. For training, an accompanying CSV file is also required, containing two columns: one for the image path and another for the label. The `utils.py` script provides tools for cropping and CSV file generation.

For more information on data loading and preprocessing, refer to the `data_loader.py` and `utils.py` scripts.

## Image Preprocessing

`main.py preprocess` command can be used to crop each images in the given folder into five sub-images. 


## Training

The model can be trained with `main.py train` 

You can specify the number of training epochs, the learning rate, and other parameters in `config.py`.

## Prediction

Use the `main.py predict` command to make predictions on new images. 

