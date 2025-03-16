# Speech Command Recognition

## Overview
This repository contains machine learning models for speech command recognition using different approaches, including logistic regression and convolutional neural networks (CNNs). The models are trained and evaluated on audio datasets to classify spoken commands accurately.

## Repository Structure

- **`LogisticRegressionModel.ipynb`** - A baseline logistic regression model for speech command recognition.
- **`TunedLogisticRegressionModel.ipynb`** - An optimized logistic regression model with hyperparameter tuning.
- **`convolutionalNeuralNetworkModel.ipynb`** - A deep learning model using CNNs to improve recognition accuracy.

## Requirements
To run the models, you need the following dependencies:

```bash
pip install numpy pandas scikit-learn librosa tensorflow keras matplotlib
```

## Usage
Clone the repository and navigate to the project directory:

```bash
git clone https://github.com/your_username/speech_command_recognition.git
cd speech_command_recognition
```

Open the Jupyter Notebooks:

```bash
jupyter notebook LogisticRegressionModel.ipynb
```

or

```bash
jupyter notebook convolutionalNeuralNetworkModel.ipynb
```

## Model Details
### 1. Logistic Regression Model
- Extracts features from speech audio.
- Uses a simple logistic regression classifier.
- Provides a baseline accuracy for speech command recognition.

### 2. Tuned Logistic Regression Model
- Implements hyperparameter tuning.
- Improves performance over the baseline logistic regression model.

### 3. Convolutional Neural Network (CNN) Model
- Uses a deep learning approach.
- Extracts features from spectrograms of audio commands.
- Achieves better performance compared to logistic regression.

## Results
The CNN model generally outperforms the logistic regression models in recognizing speech commands due to its ability to capture complex patterns in audio data.


