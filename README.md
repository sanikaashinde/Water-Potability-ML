# Water Potability Prediction Using Machine Learning

## Project Overview

This project predicts whether a water sample is potable (safe to drink) or non-potable (not safe to drink) using Machine Learning.

The project is based on the research paper:
Predicting Water Potability Using a Machine Learning Approach.

## Objective

The objective is to develop an end-to-end Machine Learning system for water potability prediction using physicochemical water-quality parameters.

The project includes data preprocessing, exploratory data analysis, model training, model evaluation, model comparison, model saving, and Streamlit deployment.

## Dataset

Dataset: Water Potability Dataset

Source: Kaggle

Dataset size:
- 3276 samples
- 9 input features
- 1 target variable

Target:
- 0 = Non-Potable / Not Safe
- 1 = Potable / Safe

## Input Features

The model uses the following parameters:

1. pH
2. Hardness
3. Solids
4. Chloramines
5. Sulfate
6. Conductivity
7. Organic Carbon
8. Trihalomethanes
9. Turbidity

## Exploratory Data Analysis

Dataset shape: 3276 rows and 10 columns.

Duplicate rows: 0

Missing values:
- pH: 491
- Sulfate: 781
- Trihalomethanes: 162

Target distribution:
- Non-Potable (0): 1998
- Potable (1): 1278

## Data Preprocessing

The following steps were performed:

1. Dataset loading
2. Duplicate checking
3. Missing-value analysis
4. Missing-value imputation
5. Train-validation-test splitting
6. Feature scaling
7. Saving the preprocessing pipeline

Dataset split:

- Training: 2293 samples
- Validation: 491 samples
- Testing: 492 samples

The preprocessing pipeline is saved as:

models/preprocessor.joblib

## Machine Learning Models

The following classification models were implemented:

- Random Forest
- Support Vector Machine (SVM)
- Logistic Regression
- Decision Tree
- K-Nearest Neighbors (KNN)

Random Forest and SVM are the primary models required by the base paper.

## Model Comparison

Validation results:

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| SVM | 62.73% | 52.08% | 52.36% | 52.22% | 0.6570 |
| Random Forest | 64.36% | 55.88% | 39.79% | 46.48% | 0.6505 |
| KNN | 63.14% | 53.79% | 37.17% | 43.96% | 0.6286 |
| Decision Tree | 60.90% | 49.70% | 43.46% | 46.37% | 0.5773 |
| Logistic Regression | 52.14% | 41.41% | 55.50% | 47.43% | 0.5331 |

## Best Model

Support Vector Machine (SVM) was selected as the final model based on validation ROC-AUC.

Validation ROC-AUC: 0.6570

The trained model is saved as:

models/best_model.joblib

## Final Test Evaluation

The selected SVM model was evaluated on 492 unseen test samples.

| Metric | Score |
|---|---:|
| Accuracy | 63.41% |
| Precision | 53.03% |
| Recall | 54.69% |
| F1-Score | 53.85% |
| ROC-AUC | 0.6540 |

Confusion Matrix:

[[207 93]
 [87 105]]

## Evaluation Outputs

The project generates:

- Confusion Matrix
- ROC Curve
- Accuracy Comparison
- F1-Score Comparison
- ROC-AUC Comparison
- Model Comparison

These files are stored in the outputs directory.

## Streamlit Dashboard

The project includes an interactive Streamlit dashboard.

Users can enter the following water-quality parameters:

- pH
- Hardness
- Solids
- Chloramines
- Sulfate
- Conductivity
- Organic Carbon
- Trihalomethanes
- Turbidity

The dashboard returns:

- SAFE TO DRINK
- NOT SAFE TO DRINK

## Project Structure

Water-Potability-ML/

    app.py
    README.md
    requirements.txt

    data/
        water_potability.csv
        processed/

    models/
        best_model.joblib
        preprocessor.joblib

    notebooks/
        01_EDA_Water_Potability.ipynb
        02_Preprocessing_and_Model_Analysis.ipynb

    src/
        preprocessing.py
        train_models.py
        evaluate_models.py
        model_comparison.py

    outputs/
        accuracy_comparison.png
        f1_comparison.png
        roc_auc_comparison.png
        model_comparison.png
        confusion_matrix.png
        roc_curve.png
        validation_results.csv
        model_evaluation.csv
        final_test_metrics.csv

## Installation

Create a virtual environment:

    python -m venv venv

Activate it on Windows PowerShell:

    .\venv\Scripts\Activate.ps1

Install dependencies:

    pip install -r requirements.txt

## Run the Streamlit Application

From the project root:

    streamlit run app.py

## Train Models

Run preprocessing:

    python .\src\preprocessing.py

Train the models:

    python .\src\train_models.py

Evaluate the selected model:

    python .\src\evaluate_models.py

Compare models:

    python .\src\model_comparison.py

## Machine Learning Workflow

Kaggle Dataset
-> Data Loading
-> Exploratory Data Analysis
-> Missing Value Handling
-> Train/Validation/Test Split
-> Imputation
-> Feature Scaling
-> Model Training
-> Model Validation
-> Model Comparison
-> Best Model Selection
-> Final Test Evaluation
-> Model Saving
-> Streamlit Prediction

## Conclusion

This project demonstrates a complete Machine Learning workflow for water potability prediction.

Multiple classification algorithms were implemented and compared. Support Vector Machine achieved the highest validation ROC-AUC and was selected as the final model.

The Streamlit application provides an interactive interface for predicting water potability from physicochemical water-quality parameters.

This project is intended for academic and research purposes. Predictions should not replace laboratory testing or official drinking-water safety standards.
