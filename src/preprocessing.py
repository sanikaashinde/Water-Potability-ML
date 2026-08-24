import os
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "water_potability.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "data",
    "processed"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


# Load Dataset
df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully")
print("Shape:", df.shape)


# Remove Duplicate Rows
duplicates = df.duplicated().sum()

print("Duplicate rows:", duplicates)

if duplicates > 0:
    df = df.drop_duplicates()
    print("Duplicates removed.")
else:
    print("No duplicates found.")


# Features and Target
FEATURES = [
    "ph",
    "Hardness",
    "Solids",
    "Chloramines",
    "Sulfate",
    "Conductivity",
    "Organic_carbon",
    "Trihalomethanes",
    "Turbidity"
]

TARGET = "Potability"

X = df[FEATURES]
y = df[TARGET]


# Train / Validation / Test Split
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)


print("\nDataset Split:")
print("Training:", X_train.shape)
print("Validation:", X_val.shape)
print("Testing:", X_test.shape)


# Numerical Preprocessing
numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_pipeline, FEATURES)
    ]
)


# Fit ONLY on Training Data
X_train_processed = preprocessor.fit_transform(X_train)
X_val_processed = preprocessor.transform(X_val)
X_test_processed = preprocessor.transform(X_test)


print("\nPreprocessing completed successfully.")
print("Processed Training Shape:", X_train_processed.shape)
print("Processed Validation Shape:", X_val_processed.shape)
print("Processed Testing Shape:", X_test_processed.shape)


# Save Processed Data
np.save(
    os.path.join(OUTPUT_DIR, "X_train.npy"),
    X_train_processed
)

np.save(
    os.path.join(OUTPUT_DIR, "X_val.npy"),
    X_val_processed
)

np.save(
    os.path.join(OUTPUT_DIR, "X_test.npy"),
    X_test_processed
)

np.save(
    os.path.join(OUTPUT_DIR, "y_train.npy"),
    y_train.to_numpy()
)

np.save(
    os.path.join(OUTPUT_DIR, "y_val.npy"),
    y_val.to_numpy()
)

np.save(
    os.path.join(OUTPUT_DIR, "y_test.npy"),
    y_test.to_numpy()
)


# Save Feature Names
with open(
    os.path.join(OUTPUT_DIR, "features.txt"),
    "w"
) as f:
    for feature in FEATURES:
        f.write(feature + "\n")


# Save Preprocessor
PREPROCESSOR_PATH = os.path.join(
    MODEL_DIR,
    "preprocessor.joblib"
)

joblib.dump(
    preprocessor,
    PREPROCESSOR_PATH
)


print("\nProcessed datasets saved to:")
print(OUTPUT_DIR)

print("\nPreprocessor saved to:")
print(PREPROCESSOR_PATH)

print("\nPreprocessing completed successfully!")
