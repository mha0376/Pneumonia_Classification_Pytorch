![banner](https://github.com/user-attachments/assets/7f6d461d-e8e6-4a85-aa6b-c04a56e13f29)

# Pneumonia Detection from Chest X-Ray Images

This repository contains a deep learning project for classifying chest X-ray images to detect pneumonia. The implementation uses PyTorch and leverages the EfficientNet-b0 architecture, pre-trained on ImageNet, fine-tuned on the Chest X-Ray Images (Pneumonia) dataset from Kaggle. The project includes data preprocessing, model training, evaluation, and visualization of results.

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

1.  **Sign up/log in to Kaggle.**
2.  **Go to your account settings and generate an API token (kaggle.json).**
3.  **Place kaggle.json in the ~/.kaggle/ directory (on Windows, this is typically C:\Users\<YourUsername>\.kaggle\).**
4.  **Run the following commands to download and unzip the dataset:**
   ```bash
    pip install torch torchvision torchmetrics matplotlib seaborn pandas numpy
   ```

## Optional: TensorBoard Setup
For real-time training visualization, install TensorBoard and configure ngrok (optional for remote access):
  ```bash
    pip install tensorboard
    wget https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.tgz
    tar xf ngrok-v3-stable-linux-amd64.tgz -C /usr/local/bin
    ngrok authtoken <your-ngrok-token>
  ```
Replace <your-ngrok-token> with your ngrok authentication token from ngrok.com.

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
The Chest X-Ray Images (Pneumonia) dataset contains labeled chest X-ray images split into:
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
