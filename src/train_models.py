import os
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

import joblib


# ============================================================
# Paths
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PROCESSED_DIR = os.path.join(
    BASE_DIR,
    "data",
    "processed"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "outputs"
)

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# Load Processed Data
# ============================================================

X_train = np.load(
    os.path.join(PROCESSED_DIR, "X_train.npy")
)

X_val = np.load(
    os.path.join(PROCESSED_DIR, "X_val.npy")
)

X_test = np.load(
    os.path.join(PROCESSED_DIR, "X_test.npy")
)

y_train = np.load(
    os.path.join(PROCESSED_DIR, "y_train.npy")
)

y_val = np.load(
    os.path.join(PROCESSED_DIR, "y_val.npy")
)

y_test = np.load(
    os.path.join(PROCESSED_DIR, "y_test.npy")
)


print("Processed data loaded successfully.")

print("Training:", X_train.shape)
print("Validation:", X_val.shape)
print("Testing:", X_test.shape)


# ============================================================
# Define Models
# ============================================================

models = {

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced"
    ),

    "SVM": SVC(
        kernel="rbf",
        probability=True,
        random_state=42,
        class_weight="balanced"
    ),

    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42
    ),

    "Decision Tree": DecisionTreeClassifier(
        random_state=42,
        class_weight="balanced"
    ),

    "KNN": KNeighborsClassifier(
        n_neighbors=5
    )
}


# ============================================================
# Evaluation Function
# ============================================================

def evaluate_model(model, X, y):

    predictions = model.predict(X)

    probabilities = model.predict_proba(X)[:, 1]

    accuracy = accuracy_score(y, predictions)

    precision = precision_score(
        y,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y,
        predictions,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y,
        probabilities
    )

    cm = confusion_matrix(
        y,
        predictions
    )

    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "ROC-AUC": roc_auc,
        "Confusion Matrix": cm
    }


# ============================================================
# Train and Validate Models
# ============================================================

validation_results = []

trained_models = {}


for name, model in models.items():

    print("\n" + "=" * 60)

    print("Training:", name)

    model.fit(
        X_train,
        y_train
    )

    trained_models[name] = model

    results = evaluate_model(
        model,
        X_val,
        y_val
    )

    validation_results.append({

        "Model": name,

        "Accuracy": results["Accuracy"],

        "Precision": results["Precision"],

        "Recall": results["Recall"],

        "F1-Score": results["F1-Score"],

        "ROC-AUC": results["ROC-AUC"]
    })

    print("Accuracy :", round(results["Accuracy"], 4))

    print("Precision:", round(results["Precision"], 4))

    print("Recall   :", round(results["Recall"], 4))

    print("F1-Score :", round(results["F1-Score"], 4))

    print("ROC-AUC  :", round(results["ROC-AUC"], 4))

    print("Confusion Matrix:")

    print(results["Confusion Matrix"])


# ============================================================
# Validation Comparison
# ============================================================

results_df = pd.DataFrame(
    validation_results
)

results_df = results_df.sort_values(
    by="ROC-AUC",
    ascending=False
)

print("\n")
print("=" * 70)
print("VALIDATION MODEL COMPARISON")
print("=" * 70)

print(
    results_df.to_string(
        index=False
    )
)


# ============================================================
# Select Best Model
# ============================================================

best_model_name = results_df.iloc[0]["Model"]

best_model = trained_models[
    best_model_name
]

print("\nBest Model:", best_model_name)


# ============================================================
# Save Validation Results
# ============================================================

results_path = os.path.join(
    OUTPUT_DIR,
    "validation_results.csv"
)

results_df.to_csv(
    results_path,
    index=False
)


# ============================================================
# Save Best Model
# ============================================================

best_model_path = os.path.join(
    MODEL_DIR,
    "best_model.joblib"
)

joblib.dump(
    best_model,
    best_model_path
)


print("\nBest model saved to:")

print(best_model_path)


# ============================================================
# Final Test Evaluation
# ============================================================

print("\n")
print("=" * 70)
print("FINAL TEST EVALUATION")
print("=" * 70)

test_results = evaluate_model(
    best_model,
    X_test,
    y_test
)

print(
    "Best Model:",
    best_model_name
)

print(
    "Accuracy :",
    round(test_results["Accuracy"], 4)
)

print(
    "Precision:",
    round(test_results["Precision"], 4)
)

print(
    "Recall   :",
    round(test_results["Recall"], 4)
)

print(
    "F1-Score :",
    round(test_results["F1-Score"], 4)
)

print(
    "ROC-AUC  :",
    round(test_results["ROC-AUC"], 4)
)

print("\nConfusion Matrix:")

print(
    test_results["Confusion Matrix"]
)


# ============================================================
# Save Final Test Metrics
# ============================================================

final_metrics = pd.DataFrame([{

    "Model": best_model_name,

    "Accuracy": test_results["Accuracy"],

    "Precision": test_results["Precision"],

    "Recall": test_results["Recall"],

    "F1-Score": test_results["F1-Score"],

    "ROC-AUC": test_results["ROC-AUC"]

}])

final_metrics_path = os.path.join(
    OUTPUT_DIR,
    "final_test_metrics.csv"
)

final_metrics.to_csv(
    final_metrics_path,
    index=False
)


print("\nFinal metrics saved to:")

print(final_metrics_path)

print("\nModel training completed successfully!")