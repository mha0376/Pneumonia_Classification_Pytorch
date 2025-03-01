![banner](https://github.com/user-attachments/assets/7f6d461d-e8e6-4a85-aa6b-c04a56e13f29)

# Pneumonia Detection from Chest X-Ray Images

This repository contains a deep learning project for classifying chest X-ray images to detect pneumonia. The implementation uses PyTorch and leverages the EfficientNet-b0 architecture, pre-trained on ImageNet, fine-tuned on the [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
 dataset from Kaggle. The project includes data preprocessing, model training, evaluation, and visualization of results.

## Table of Contents
*   Installation
*   Usage
*   Dataset
*   Model Architecture
*   Training and Evaluation
*   Results
*   Contributing

## Installation
To run this project, you'll need Python 3.x installed. Follow these steps to set up the environment:

## Prerequisites
*   Python 3.x
*   pip (Python package manager)
*   A Kaggle account and API token for dataset access

## Install Dependencies
Install the required Python packages using pip:

  ```bash
  pip install torch torchvision torchmetrics matplotlib seaborn pandas numpy
  ```

## Download the Dataset
The project uses the Kaggle API to download the dataset. Set up your Kaggle API token as follows:

1.  **Sign up/log in to [Kaggle](https://www.kaggle.com/).**
2.  **Go to your account settings and generate an API token (kaggle.json).**
3.  **Place kaggle.json in the ~/.kaggle/ directory (on Windows, this is typically C:\Users\<YourUsername>\.kaggle\).**
4.  **Run the following commands to download and unzip the dataset:**
   ```bash
   kaggle datasets download paultimothymooney/chest-xray-pneumonia
   unzip chest-xray-pneumonia.zip
   ```

## Optional: TensorBoard Setup
For real-time training visualization, install TensorBoard and configure ngrok (optional for remote access):
  ```bash
  pip install tensorboard
  wget https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.tgz
  tar xf ngrok-v3-stable-linux-amd64.tgz -C /usr/local/bin
  ngrok authtoken <your-ngrok-token>
  ```
Replace <your-ngrok-token> with your ngrok authentication token from [ngrok.com](https://ngrok.com/).

## Usage
The code is provided as a Jupyter notebook (.ipynb). To execute it:
1.  **Ensure all dependencies are installed and the dataset is downloaded.**
2.  **Open the notebook in Jupyter:**
   ```bash
   jupyter notebook
   ```
3.  **Run the cells sequentially to:**
  *   Install dependencies.
  *   Download and preprocess the dataset.
  *   Define and train the model.
  *   Evaluate performance and visualize results.
    
For optimal performance, use a GPU-enabled environment (e.g., Google Colab with GPU runtime or a local machine with CUDA). The code automatically detects and uses CUDA if available.

## Dataset
The [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia) dataset contains labeled chest X-ray images split into:
*   Training Set: Used to train the model.
*   Validation Set: Used to tune hyperparameters and monitor training.
*   Test Set: Used to evaluate final model performance.
  
The dataset is organized into subdirectories (NORMAL and PNEUMONIA) under chest_xray/train, chest_xray/val, and chest_xray/test. Images are preprocessed with resizing, normalization, and data augmentation (random horizontal flips and rotations) for training.

## Model Architecture
The model is based on EfficientNet-b0, a state-of-the-art convolutional neural network (CNN) pre-trained on ImageNet. It has been fine-tuned for binary classification (normal vs. pneumonia) by modifying the classifier head:

```bash
model = torchvision.models.efficientnet_b0(pretrained=True)
model.classifier = nn.Sequential(
nn.Dropout(p=0.2, inplace=True),
nn.Linear(in_features=1280, out_features=2)
)
```
*   Input: 224x224 RGB images.
*   Output: 2 classes (0 = Normal, 1 = Pneumonia).
*   Parameters: Approximately 4 million trainable parameters.

## Training and Evaluation
### Training
*   Optimizer: Adam with a learning rate of 0.001.
*   Loss Function: Cross-Entropy Loss.
*   Scheduler: ReduceLROnPlateau (reduces learning rate on validation loss plateau).
*   Early Stopping: Stops training if validation loss doesn't improve for 5 epochs.
*   Epochs: Configurable (default: 10).
*   Batch Size: 64.
  
Training metrics (loss, accuracy, precision, recall, F1 score) are logged using TensorBoard.

## Evaluation
The model is evaluated on the test set after training, with the best-performing model (based on validation accuracy) loaded from checkpoints. Key metrics include:
*   Accuracy
*   Precision
*   Recall
*   F1 Score
*   Confusion Matrix
  
Results are saved to test_results.txt, and plots (loss/accuracy over epochs, confusion matrix) are saved as .png files.

## Results
After training, the notebook generates:
*   Plots:
    *   Training and validation loss/accuracy over epochs (training_metrics.png).
    *   Confusion matrix for the test set (confusion_matrix.png).
*   Text File: Test metrics (test_results.txt) including the best validation accuracy and epoch.
  
To view training progress in real-time:
```bash
tensorboard --logdir ./runs/
```
Access the TensorBoard URL locally or via ngrok if configured.

Sample output files are not included in the repository but will be generated upon running the notebook.

## Contributing
Contributions are welcome! Please:
1.  **Fork the repository.**
2.  **Create a new branch for your feature or bug fix.**
3.  **Submit a pull request with a clear description of your changes.**
   
For major changes, open an issue first to discuss your ideas.







# Pneumonia Classification with PyTorch

## Project Description

This project implements a Pneumonia classification model using PyTorch.  The primary goal is to classify chest X-ray images as either having pneumonia or not.  While a detailed description was not initially provided, the provided files suggest a straightforward training and evaluation pipeline using PyTorch.  Further analysis of the code is needed to determine the specific model architecture and datasets used.  This README will evolve as more details are uncovered.

## Features and Functionality

Based on the file structure, the anticipated features and functionality are:

*   **Image Classification:** Classifies chest X-ray images into 'Pneumonia' or 'Normal' categories.
*   **Model Training:** Trains a convolutional neural network (CNN) model using PyTorch.
*   **Model Evaluation:** Evaluates the trained model on a test dataset.
*   **Data Loading and Preprocessing:** Loads and preprocesses image data for training and evaluation.  Details on specific preprocessing steps are currently unavailable without examining the data loading script.
*   **(Potentially) Model Saving and Loading:** Likely includes functionality to save and load trained model weights.

## Technology Stack

*   **Python:** Primary programming language.
*   **PyTorch:** Deep learning framework for model building, training, and evaluation.
*   **NumPy:** Library for numerical computations.
*   **PIL (Pillow):** Python Imaging Library for image manipulation.
*   **Other potential libraries:** Matplotlib (for visualization), Scikit-learn (for metrics).

## Prerequisites

Before running the code, ensure you have the following installed:

1.  **Python:**  Version 3.7 or higher is recommended.
2.  **PyTorch:** Install using pip or conda.  Choose the appropriate version based on your system and CUDA compatibility.  Example using pip:

    ```bash
    pip install torch torchvision torchaudio
    ```

    Refer to the official PyTorch website ([https://pytorch.org/](https://pytorch.org/)) for detailed installation instructions.
3.  **NumPy:**

    ```bash
    pip install numpy
    ```
4.  **Pillow:**

    ```bash
    pip install Pillow
    ```
5. **CUDA (Optional):** If you have a CUDA-enabled GPU, install the CUDA toolkit and cuDNN to accelerate training.  This requires configuring PyTorch to use your GPU.

## Installation Instructions

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/mha0376/Pneumonia_Classification_Pytorch.git
    cd Pneumonia_Classification_Pytorch
    ```

2.  **(Optional) Create a virtual environment:**  It is highly recommended to create a virtual environment to isolate project dependencies.

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Linux/macOS
    venv\Scripts\activate  # On Windows
    ```

3.  **Install the required packages:** (If not already installed in prerequisites)

    ```bash
    pip install torch torchvision torchaudio numpy Pillow
    ```

## Usage Guide

A detailed usage guide depends on the entry point of the application (e.g., a `train.py` or `main.py` script). Assuming a typical training script named `train.py`, the following steps outline the general usage:

1.  **Data Preparation:**  The code likely expects the data to be organized in a specific directory structure.  Examine the data loading part of the training script (e.g., `train.py`) to understand the expected format.  Typically, this might involve separate directories for training and validation data, with subdirectories for each class (Pneumonia and Normal).  An example file structure might look like this:

    ```
    data/
        train/
            Pneumonia/
                image1.jpg
                image2.png
                ...
            Normal/
                image1.jpg
                image2.png
                ...
        val/
            Pneumonia/
                image1.jpg
                image2.png
                ...
            Normal/
                image1.jpg
                image2.png
                ...
    ```

2.  **Training the Model:** Run the training script. This might involve specifying command-line arguments for hyperparameters such as learning rate, batch size, and number of epochs.

    ```bash
    python train.py --learning_rate 0.001 --batch_size 32 --epochs 10
    ```

    **(Note:**  Replace `train.py` and the arguments with the actual script name and arguments used in the repository. Without access to the files, specific arguments cannot be provided.  Check for a `train.py` or similar script that handles the training loop.)

3.  **Evaluating the Model:** After training, evaluate the model on a test dataset.  This might involve a separate evaluation script or be integrated into the training script.

    ```bash
    python evaluate.py --model_path path/to/your/trained_model.pth --test_data_dir path/to/your/test_data
    ```

    **(Note:**  Replace `evaluate.py`, `--model_path`, and `--test_data_dir` with the appropriate script and arguments.)

4. **Configuration Files:** Look for configuration files (e.g., `config.yaml` or `config.json`) or command-line arguments that define the model architecture, dataset paths, and training parameters. If a config file is found, examine its contents to understand configurable options.

## API Documentation

This project, in its current described state, is unlikely to have a formal API. The primary interaction is through running training and evaluation scripts. If a more complex system with callable functions or classes exists, this section will be updated with relevant documentation.

## Contributing Guidelines

Contributions are welcome! To contribute to this project, follow these steps:

1.  **Fork the repository.**
2.  **Create a new branch for your feature or bug fix:**

    ```bash
    git checkout -b feature/your-feature-name
    ```

3.  **Make your changes and commit them with descriptive messages.**
4.  **Push your changes to your forked repository.**
5.  **Submit a pull request to the `main` branch of the original repository.**

Please ensure that your code adheres to the project's coding style and includes appropriate tests.

## License Information

No license information was provided. By default, without a license, the code is considered to have all rights reserved.  The owner of the repository needs to explicitly include a license file (e.g., `LICENSE.txt` with content such as the MIT License or Apache 2.0 License) to allow others to use, modify, or distribute the code.

## Contact/Support Information

For questions or support, please contact the repository owner through GitHub.  (No specific contact information was provided, so users will need to rely on GitHub's issue tracking or discussion features).
