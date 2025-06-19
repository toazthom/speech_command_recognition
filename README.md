# Speech Command Recognition

## Overview
This repository contains machine learning models for speech command recognition using different approaches, including logistic regression, SVM, and convolutional neural networks (CNNs). The models are trained and evaluated on audio datasets to classify spoken commands accurately.

## Repository Structure

- **`LogisticRegressionModel.ipynb`** – Baseline model using logistic regression.
- **`TunedLogisticRegressionModel.ipynb`** – Tuned logistic regression model with hyperparameter optimization and SMOTE-based class balancing.
- **`convolutionalNeuralNetworkModel.ipynb`** – Convolutional Neural Network (CNN) model for improved accuracy.
- **`SupportVectorMachineModel.ipynb`** – SVM model with RBF kernel, hyperparameter tuning via grid search.
- **`SpeechRecognitionLongPaper.pdf`** – Final research paper describing the project.


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
- Uses MFCC features extracted from speech audio as input.
- Implements a basic logistic regression classifier using Scikit-learn.
- Serves as a baseline model for comparison with more complex methods.
- Achieved ~61% accuracy without tuning or feature reduction.

### 2. Tuned Logistic Regression Model
- Adds preprocessing: feature scaling with `StandardScaler` and class balancing with SMOTE.
- Uses `RandomizedSearchCV` for hyperparameter tuning (solver, penalty, regularization strength).
- Achieved a modest accuracy improvement (~65%) and better class prediction balance.
- Performance was slightly improved with PCA feature reduction.

### 3. Convolutional Neural Network (CNN) Model
- Treats MFCC arrays as 2D input for convolutional processing.
- Composed of convolutional layers, ReLU activations, a dense layer, and softmax output.
- Trained using the Adam optimizer and sparse categorical cross-entropy loss.
- Achieved 83% accuracy on raw MFCCs and 88% accuracy with PCA-applied features.
- Showed the largest performance boost from dimensionality reduction.

### 4. Support Vector Machine (SVM) Model
- Uses RBF (Radial Basis Function) kernel to capture non-linear relationships in MFCC data.
- Preprocessed using scaling and optional PCA.
- Tuned using `GridSearchCV` over `C` and `gamma` values.
- Achieved 88% accuracy both before and after PCA.
- Demonstrated strong generalization and consistent performance across all evaluated metrics.

## Results
The CNN and SVM models generally outperforms the logistic regression models in recognizing speech commands due to their abilities to capture complex patterns in audio data. Use of PCA reduction helped the logistic regression and CNN models, but not the SVM model.


