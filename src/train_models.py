import os
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)


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
# Evaluation Function
# ============================================================

def evaluate_model(model, X, y):

    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]

    return {
        "Accuracy": accuracy_score(y, predictions),
        "Precision": precision_score(
            y, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y, predictions, zero_division=0
        ),
        "F1-Score": f1_score(
            y, predictions, zero_division=0
        ),
        "ROC-AUC": roc_auc_score(
            y, probabilities
        ),
        "Confusion Matrix": confusion_matrix(
            y, predictions
        )
    }


# ============================================================
# Base Models
# ============================================================

models = {

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
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
# Train Base Models
# ============================================================

validation_results = []
trained_models = {}

for name, model in models.items():

    print("\n" + "=" * 70)
    print("Training:", name)
    print("=" * 70)

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
# Proposed Model
# Hyperparameter-Tuned Random Forest
# ============================================================

print("\n")
print("=" * 70)
print("PROPOSED MODEL")
print("Hyperparameter-Tuned Random Forest")
print("=" * 70)

rf = RandomForestClassifier(
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

param_grid = {

    "n_estimators": [200, 300],

    "max_depth": [None, 10, 20],

    "min_samples_split": [2, 5],

    "min_samples_leaf": [1, 2],

    "max_features": ["sqrt", "log2"]
}


grid_search = GridSearchCV(

    estimator=rf,

    param_grid=param_grid,

    scoring="roc_auc",

    cv=5,

    n_jobs=-1,

    verbose=1
)


grid_search.fit(
    X_train,
    y_train
)

proposed_model = grid_search.best_estimator_

trained_models[
    "Proposed Model - Tuned Random Forest"
] = proposed_model


print("\nBest Proposed Model Parameters:")

print(
    grid_search.best_params_
)

print(
    "Best Cross-Validation ROC-AUC:",
    round(grid_search.best_score_, 4)
)


# ============================================================
# Proposed Model Validation
# ============================================================

proposed_results = evaluate_model(
    proposed_model,
    X_val,
    y_val
)

validation_results.append({

    "Model": "Proposed Model - Tuned Random Forest",

    "Accuracy": proposed_results["Accuracy"],

    "Precision": proposed_results["Precision"],

    "Recall": proposed_results["Recall"],

    "F1-Score": proposed_results["F1-Score"],

    "ROC-AUC": proposed_results["ROC-AUC"]
})


print("\nProposed Model Validation Results")

print(
    "Accuracy :",
    round(proposed_results["Accuracy"], 4)
)

print(
    "Precision:",
    round(proposed_results["Precision"], 4)
)

print(
    "Recall   :",
    round(proposed_results["Recall"], 4)
)

print(
    "F1-Score :",
    round(proposed_results["F1-Score"], 4)
)

print(
    "ROC-AUC  :",
    round(proposed_results["ROC-AUC"], 4)
)

print("\nConfusion Matrix:")
print(proposed_results["Confusion Matrix"])


# ============================================================
# Cross-Validation for All Models
# ============================================================

print("\n")
print("=" * 70)
print("5-FOLD CROSS-VALIDATION")
print("=" * 70)

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

cv_results = []

for name, model in trained_models.items():

    print("\nCross-validating:", name)

    scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1
    )

    cv_results.append({

        "Model": name,

        "CV_ROC_AUC_Mean": scores.mean(),

        "CV_ROC_AUC_Std": scores.std()
    })

    print(
        "Fold Scores:",
        np.round(scores, 4)
    )

    print(
        "Mean ROC-AUC:",
        round(scores.mean(), 4)
    )

    print(
        "Std:",
        round(scores.std(), 4)
    )


cv_df = pd.DataFrame(
    cv_results
)

cv_path = os.path.join(
    OUTPUT_DIR,
    "cross_validation_results.csv"
)

cv_df.to_csv(
    cv_path,
    index=False
)

print("\nCross-validation results saved:")
print(cv_path)


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
print("FINAL VALIDATION MODEL COMPARISON")
print("=" * 70)

print(
    results_df.to_string(
        index=False
    )
)


results_path = os.path.join(
    OUTPUT_DIR,
    "validation_results.csv"
)

results_df.to_csv(
    results_path,
    index=False
)


# ============================================================
# Select Best Model
# ============================================================

best_model_name = results_df.iloc[0]["Model"]

best_model = trained_models[
    best_model_name
]

print("\n")
print("=" * 70)
print("BEST MODEL SELECTION")
print("=" * 70)

print(
    "Best Model:",
    best_model_name
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


# ============================================================
# Save Final Confusion Matrix
# ============================================================

cm = test_results["Confusion Matrix"]

cm_df = pd.DataFrame(
    cm,
    index=["Actual Non-Potable", "Actual Potable"],
    columns=["Predicted Non-Potable", "Predicted Potable"]
)

cm_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "final_confusion_matrix.csv"
    )
)

# ============================================================
# Save Final Confusion Matrix Plot
# ============================================================

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve

cm_display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Non-Potable", "Potable"]
)

fig, ax = plt.subplots(figsize=(8, 6))
cm_display.plot(ax=ax)

ax.set_title(
    f"Confusion Matrix - {best_model_name}"
)

fig.tight_layout()

fig.savefig(
    os.path.join(
        OUTPUT_DIR,
        "confusion_matrix.png"
    ),
    dpi=200,
    bbox_inches="tight"
)

plt.close(fig)


# ============================================================
# Save Final ROC Curve
# ============================================================

if hasattr(best_model, "predict_proba"):

    test_probabilities = best_model.predict_proba(X_test)[:, 1]

    fpr, tpr, _ = roc_curve(
        y_test,
        test_probabilities
    )

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(
        fpr,
        tpr,
        label=f"ROC-AUC = {test_results['ROC-AUC']:.4f}"
    )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--"
    )

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")

    ax.set_title(
        f"ROC Curve - {best_model_name}"
    )

    ax.legend(loc="lower right")

    fig.tight_layout()

    fig.savefig(
        os.path.join(
            OUTPUT_DIR,
            "roc_curve.png"
        ),
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)

print("\nFinal graphs saved:")
print(os.path.join(OUTPUT_DIR, "confusion_matrix.png"))
print(os.path.join(OUTPUT_DIR, "roc_curve.png"))



print("\nFinal metrics saved to:")
print(final_metrics_path)

print("\n")
print("=" * 70)
print("MODEL TRAINING COMPLETED SUCCESSFULLY")
print("=" * 70)
