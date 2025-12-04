# Fashion MNIST Classification with PyTorch

This project implements a Convolutional Neural Network (CNN) using PyTorch to classify images from the Fashion MNIST dataset. The dataset consists of 60,000 training images and 10,000 test images of 10 different fashion categories.

## 📋 Project Overview

The notebook performs the following steps:
1.  **Data Loading**: Loads the Fashion MNIST dataset from a CSV file.
2.  **Data Preprocessing**:
    * Splits the data into training and testing sets.
    * Normalizes pixel values to the range [0, 1].
    * Converts data into PyTorch tensors.
    * Creates custom `Dataset` and `DataLoader` objects for batch processing.
3.  **Model Architecture**: Defines a CNN (`MyNN`) with:
    * Two convolutional layers with ReLU activation and Batch Normalization.
    * Max pooling layers.
    * Flattening layer.
    * Fully connected (Dense) layers with ReLU activation and Dropout for regularization.
    * Output layer with 10 units (for 10 classes).
4.  **Training**:
    * Uses Cross-Entropy Loss and SGD optimizer.
    * Trains the model for 100 epochs on a GPU (if available).
    * Tracks and prints the average loss per epoch.
5.  **Evaluation**:
    * Evaluates the trained model on both the test and training datasets.
    * Calculates and prints the classification accuracy.

## 🛠️ Technologies Used

* **Python**
* **PyTorch** (Neural Network framework)
* **Pandas** (Data manipulation)
* **Scikit-learn** (Data splitting)
* **Matplotlib** (Visualization)

## 📊 Model Performance

* **Test Accuracy**: ~92.6%
* **Training Accuracy**: ~99.9%

The model demonstrates high accuracy on the training set and good generalization to the test set, effectively classifying fashion items.

## 🚀 Getting Started

1.  **Prerequisites**:
    * Python 3.x
    * Install required libraries:
        ```bash
        pip install torch torchvision pandas scikit-learn matplotlib
        ```

2.  **Dataset**:
    * Ensure you have the `fashion-mnist_train.csv` file. The code assumes it is located at `/content/fashion-mnist_train.csv` (typical for Google Colab). You may need to adjust the path if running locally.

3.  **Running the Code**:
    * Open the Jupyter Notebook (`fashion_mnist_pytorch.ipynb`).
    * Run the cells sequentially to load data, train the model, and see the results.

## 🧠 Neural Network Architecture

```python
class MyNN(nn.Module):
    def __init__(self, input_features):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_features, 32, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*7*7, 128),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(64, 10)
        )# pytorch_m
