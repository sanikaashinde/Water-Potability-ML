import os
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "outputs"
)

RESULTS_PATH = os.path.join(
    OUTPUT_DIR,
    "validation_results.csv"
)

df = pd.read_csv(RESULTS_PATH)

metrics = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1-Score",
    "ROC-AUC"
]

print("=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

print("\nValidation Results:")
print(df.to_string(index=False))


# ------------------------------------------------------------
# Accuracy Comparison
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))

plt.bar(
    df["Model"],
    df["Accuracy"]
)

plt.title("Model Accuracy Comparison")
plt.xlabel("Model")
plt.ylabel("Accuracy")

plt.xticks(
    rotation=30,
    ha="right"
)

plt.ylim(0, 1)

plt.tight_layout()

accuracy_path = os.path.join(
    OUTPUT_DIR,
    "accuracy_comparison.png"
)

plt.savefig(
    accuracy_path,
    dpi=300
)

plt.close()


# ------------------------------------------------------------
# F1 Score Comparison
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))

plt.bar(
    df["Model"],
    df["F1-Score"]
)

plt.title("Model F1-Score Comparison")
plt.xlabel("Model")
plt.ylabel("F1-Score")

plt.xticks(
    rotation=30,
    ha="right"
)

plt.ylim(0, 1)

plt.tight_layout()

f1_path = os.path.join(
    OUTPUT_DIR,
    "f1_comparison.png"
)

plt.savefig(
    f1_path,
    dpi=300
)

plt.close()


# ------------------------------------------------------------
# ROC-AUC Comparison
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))

plt.bar(
    df["Model"],
    df["ROC-AUC"]
)

plt.title("Model ROC-AUC Comparison")
plt.xlabel("Model")
plt.ylabel("ROC-AUC")

plt.xticks(
    rotation=30,
    ha="right"
)

plt.ylim(0, 1)

plt.tight_layout()

roc_path = os.path.join(
    OUTPUT_DIR,
    "roc_auc_comparison.png"
)

plt.savefig(
    roc_path,
    dpi=300
)

plt.close()


# ------------------------------------------------------------
# Combined Metrics
# ------------------------------------------------------------

plt.figure(figsize=(12, 7))

x = range(len(df))

width = 0.15

for i, metric in enumerate(metrics):

    positions = [
        value + (i - 2) * width
        for value in x
    ]

    plt.bar(
        positions,
        df[metric],
        width=width,
        label=metric
    )

plt.xticks(
    list(x),
    df["Model"],
    rotation=30,
    ha="right"
)

plt.ylabel("Score")

plt.title("Machine Learning Model Performance Comparison")

plt.ylim(0, 1)

plt.legend()

plt.tight_layout()

combined_path = os.path.join(
    OUTPUT_DIR,
    "model_comparison.png"
)

plt.savefig(
    combined_path,
    dpi=300
)

plt.close()


print("\nGraphs saved successfully:")

print(accuracy_path)
print(f1_path)
print(roc_path)
print(combined_path)

print("\n" + "=" * 70)
print("MODEL COMPARISON COMPLETED")
print("=" * 70)
