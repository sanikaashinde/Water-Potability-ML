# Water Potability Prediction Using Machine Learning

## Project Overview

This project predicts whether a water sample is potable (safe to drink) or non-potable (not safe to drink) using Machine Learning.

The project is based on the research paper:

**"Predicting Water Potability Using a Machine Learning Approach"**

The system implements multiple classification algorithms, compares their performance using standard evaluation metrics, and introduces a **Hyperparameter-Tuned Random Forest** as the proposed model.

An interactive **Streamlit dashboard** is also provided for water potability prediction.

---

## Objective

The main objectives of this project are:

- Analyze water-quality parameters using Exploratory Data Analysis.
- Handle missing values and preprocess the dataset.
- Split the dataset into training, validation, and testing sets.
- Train multiple Machine Learning classification models.
- Compare model performance using Accuracy, Precision, Recall, F1-Score, and ROC-AUC.
- Perform 5-Fold Cross-Validation.
- Develop a proposed model using hyperparameter tuning.
- Evaluate the proposed model on unseen test data.
- Deploy the prediction system using Streamlit.

---

## Dataset

**Dataset:** Water Potability Dataset  
**Source:** Kaggle

### Dataset Size

- 3276 samples
- 9 input features
- 1 target variable

### Target Variable

| Value | Meaning |
|---|---|
| 0 | Non-Potable / Not Safe |
| 1 | Potable / Safe |

---

## Input Features

The model uses the following physicochemical water-quality parameters:

1. pH
2. Hardness
3. Solids
4. Chloramines
5. Sulfate
6. Conductivity
7. Organic Carbon
8. Trihalomethanes
9. Turbidity

---

## Exploratory Data Analysis

The dataset contains **3276 rows and 10 columns**.

### Duplicate Records

- Duplicate rows: **0**

### Missing Values

| Feature | Missing Values |
|---|---:|
| pH | 491 |
| Sulfate | 781 |
| Trihalomethanes | 162 |

### Target Distribution

| Class | Samples |
|---|---:|
| Non-Potable (0) | 1998 |
| Potable (1) | 1278 |

---

## Data Preprocessing

The following preprocessing steps were performed:

1. Dataset loading
2. Duplicate checking
3. Missing-value analysis
4. Median-value imputation
5. Train-validation-test splitting
6. Feature scaling using StandardScaler
7. Preprocessing pipeline creation
8. Saving the preprocessing pipeline using Joblib

### Dataset Split

| Dataset | Samples |
|---|---:|
| Training | 2293 |
| Validation | 491 |
| Testing | 492 |

The preprocessing pipeline is saved as:

`models/preprocessor.joblib`

---

## Machine Learning Models

The following classification models were implemented:

- Random Forest
- Support Vector Machine (SVM)
- Logistic Regression
- Decision Tree
- K-Nearest Neighbors (KNN)

Random Forest and SVM were included as the primary models based on the base research paper.

---

## Proposed Model

The proposed model in this project is:

### Hyperparameter-Tuned Random Forest

Random Forest hyperparameters were optimized using **GridSearchCV with 5-fold cross-validation**.

The search considered:

- Number of estimators
- Maximum depth
- Maximum features
- Minimum samples split
- Minimum samples leaf

### Best Parameters

```text
n_estimators = 300
max_depth = 10
max_features = sqrt
min_samples_split = 5
min_samples_leaf = 1