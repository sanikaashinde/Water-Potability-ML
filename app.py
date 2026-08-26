import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import matplotlib.pyplot as plt

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Water Potability Prediction",
    page_icon=None,
    layout="wide"
)

# --------------------------------------------------
# PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "water_potability.csv"
MODEL_PATH = BASE_DIR / "models" / "best_model.joblib"
PREPROCESSOR_PATH = BASE_DIR / "models" / "preprocessor.joblib"

OUTPUTS_DIR = BASE_DIR / "outputs"

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

@st.cache_data
def load_dataset():
    return pd.read_csv(DATA_PATH)

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

# --------------------------------------------------
# LOAD PREPROCESSOR
# --------------------------------------------------

@st.cache_resource
def load_preprocessor():
    return joblib.load(PREPROCESSOR_PATH)

try:
    df = load_dataset()
    model = load_model()
    preprocessor = load_preprocessor()
except Exception as e:
    st.error("Unable to load project files.")
    st.exception(e)
    st.stop()

# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title("Water Potability Prediction")
st.caption("Machine Learning based water quality classification system")

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Page",
    [
        "Dashboard",
        "Dataset Summary",
        "Model Performance",
        "Model Comparison",
        "Water Potability Prediction"
    ]
)

# --------------------------------------------------
# DASHBOARD
# --------------------------------------------------

if page == "Dashboard":

    st.header("Project Overview")

    st.write(
        "This project predicts whether a water sample is potable "
        "or non-potable using physicochemical water-quality parameters."
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Dataset Samples", len(df))

    with col2:
        st.metric("Input Features", 9)

    with col3:
        st.metric("Best Model", "Tuned Random Forest")

    with col4:
        st.metric("Target Classes", 2)

    st.divider()

    st.subheader("Project Information")

    info_col1, info_col2 = st.columns(2)

    with info_col1:
        st.write("Project: Water Potability Prediction")
        st.write("Dataset: Kaggle Water Potability Dataset")
        st.write("Primary Models: Random Forest and SVM")
        st.write("Best Model: Hyperparameter-Tuned Random Forest")

    with info_col2:
        st.write("Target: Potability")
        st.write("0 = Non-Potable / Not Safe")
        st.write("1 = Potable / Safe")
        st.write("Total Samples: 3276")

    st.divider()

    st.subheader("Machine Learning Workflow")

    st.write(
        "Dataset Collection -> EDA -> Missing Value Handling -> "
        "Train/Validation/Test Split -> Imputation -> Feature Scaling -> "
        "Model Training -> Model Evaluation -> Model Comparison -> "
        "Best Model Selection -> Final Prediction"
    )

# --------------------------------------------------
# DATASET SUMMARY
# --------------------------------------------------

elif page == "Dataset Summary":

    st.header("Dataset Summary")

    st.subheader("Dataset Preview")

    st.dataframe(
        df.head(10),
        use_container_width=True
    )

    st.subheader("Dataset Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Rows", df.shape[0])

    with col2:
        st.metric("Columns", df.shape[1])

    with col3:
        st.metric("Duplicate Rows", int(df.duplicated().sum()))

    st.subheader("Missing Values")

    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)

    if len(missing) > 0:
        missing_df = pd.DataFrame({
            "Feature": missing.index,
            "Missing Values": missing.values
        })

        st.dataframe(
            missing_df,
            use_container_width=True,
            hide_index=True
        )
    else:
        st.success("No missing values found.")

    st.subheader("Target Distribution")

    target_counts = df["Potability"].value_counts().sort_index()

    target_df = pd.DataFrame({
        "Class": ["Non-Potable", "Potable"],
        "Count": [
            int(target_counts.get(0, 0)),
            int(target_counts.get(1, 0))
        ]
    })

    st.dataframe(
        target_df,
        use_container_width=True,
        hide_index=True
    )

    fig, ax = plt.subplots()

    ax.bar(
        ["Non-Potable", "Potable"],
        target_df["Count"]
    )

    ax.set_title("Potability Class Distribution")
    ax.set_xlabel("Water Class")
    ax.set_ylabel("Number of Samples")

    st.pyplot(fig)

    plt.close(fig)

# --------------------------------------------------
# MODEL PERFORMANCE
# --------------------------------------------------

elif page == "Model Performance":

    st.header("Final Model Evaluation")

    st.write(
        "The Support Vector Machine model was selected as the final "
        "model based on validation ROC-AUC."
    )

    metrics_path = OUTPUTS_DIR / "model_evaluation.csv"

    if metrics_path.exists():

        metrics_df = pd.read_csv(metrics_path)

        st.subheader("Final Test Metrics")

        st.dataframe(
            metrics_df,
            use_container_width=True,
            hide_index=True
        )

    else:

        metrics_path = OUTPUTS_DIR / "final_test_metrics.csv"

        if metrics_path.exists():

            metrics_df = pd.read_csv(metrics_path)

            st.dataframe(
                metrics_df,
                use_container_width=True,
                hide_index=True
            )

        else:
            st.warning("Evaluation metrics file not found.")

    st.subheader("Confusion Matrix")

    confusion_path = OUTPUTS_DIR / "confusion_matrix.png"

    if confusion_path.exists():
        st.image(
            str(confusion_path),
            caption="Tuned Random Forest Confusion Matrix",
            use_container_width=True
        )
    else:
        st.warning("Confusion matrix image not found.")

    st.subheader("ROC Curve")

    roc_path = OUTPUTS_DIR / "roc_curve.png"

    if roc_path.exists():
        st.image(
            str(roc_path),
            caption="Tuned Random Forest ROC Curve",
            use_container_width=True
        )
    else:
        st.warning("ROC curve image not found.")

# --------------------------------------------------
# MODEL COMPARISON
# --------------------------------------------------

elif page == "Model Comparison":

    st.header("Model Comparison")

    validation_path = OUTPUTS_DIR / "validation_results.csv"

    if validation_path.exists():

        comparison_df = pd.read_csv(validation_path)

        st.subheader("Validation Results")

        st.dataframe(
            comparison_df,
            use_container_width=True,
            hide_index=True
        )

    else:
        st.warning("Validation results file not found.")

    st.subheader("Accuracy Comparison")

    accuracy_path = OUTPUTS_DIR / "accuracy_comparison.png"

    if accuracy_path.exists():
        st.image(
            str(accuracy_path),
            caption="Model Accuracy Comparison",
            use_container_width=True
        )

    st.subheader("F1-Score Comparison")

    f1_path = OUTPUTS_DIR / "f1_comparison.png"

    if f1_path.exists():
        st.image(
            str(f1_path),
            caption="Model F1-Score Comparison",
            use_container_width=True
        )

    st.subheader("ROC-AUC Comparison")

    roc_auc_path = OUTPUTS_DIR / "roc_auc_comparison.png"

    if roc_auc_path.exists():
        st.image(
            str(roc_auc_path),
            caption="Model ROC-AUC Comparison",
            use_container_width=True
        )

    st.subheader("Overall Model Comparison")

    comparison_image = OUTPUTS_DIR / "model_comparison.png"

    if comparison_image.exists():
        st.image(
            str(comparison_image),
            caption="Overall Model Comparison",
            use_container_width=True
        )

# --------------------------------------------------
# WATER POTABILITY PREDICTION
# --------------------------------------------------

elif page == "Water Potability Prediction":

    st.header("Water Potability Prediction")

    st.write(
        "Enter the physicochemical properties of the water sample "
        "to predict whether the water is potable."
    )

    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:

        ph = st.number_input(
            "pH",
            min_value=0.0,
            max_value=14.0,
            value=7.0,
            step=0.01
        )

        hardness = st.number_input(
            "Hardness",
            min_value=0.0,
            value=196.0,
            step=0.01
        )

        solids = st.number_input(
            "Solids",
            min_value=0.0,
            value=22000.0,
            step=0.01
        )

    with col2:

        chloramines = st.number_input(
            "Chloramines",
            min_value=0.0,
            value=7.0,
            step=0.01
        )

        sulfate = st.number_input(
            "Sulfate",
            min_value=0.0,
            value=333.0,
            step=0.01
        )

        conductivity = st.number_input(
            "Conductivity",
            min_value=0.0,
            value=426.0,
            step=0.01
        )

    with col3:

        organic_carbon = st.number_input(
            "Organic Carbon",
            min_value=0.0,
            value=14.0,
            step=0.01
        )

        trihalomethanes = st.number_input(
            "Trihalomethanes",
            min_value=0.0,
            value=66.0,
            step=0.01
        )

        turbidity = st.number_input(
            "Turbidity",
            min_value=0.0,
            value=4.0,
            step=0.01
        )

    st.divider()

    predict_button = st.button(
        "Predict Water Potability",
        type="primary",
        use_container_width=True
    )

    if predict_button:

        input_data = pd.DataFrame([{
            "ph": ph,
            "Hardness": hardness,
            "Solids": solids,
            "Chloramines": chloramines,
            "Sulfate": sulfate,
            "Conductivity": conductivity,
            "Organic_carbon": organic_carbon,
            "Trihalomethanes": trihalomethanes,
            "Turbidity": turbidity
        }])

        try:

            processed_input = preprocessor.transform(input_data)

            prediction = model.predict(processed_input)[0]

            st.divider()

            st.subheader("Prediction Result")

            if int(prediction) == 1:

                st.success("SAFE TO DRINK")

                st.write(
                    "The model predicts that this water sample is potable."
                )

            else:

                st.error("NOT SAFE TO DRINK")

                st.write(
                    "The model predicts that this water sample is non-potable."
                )

            if hasattr(model, "predict_proba"):

                probability = model.predict_proba(
                    processed_input
                )[0]

                potable_probability = probability[1] * 100
                non_potable_probability = probability[0] * 100

                st.subheader("Prediction Probability")

                prob_col1, prob_col2 = st.columns(2)

                with prob_col1:
                    st.metric(
                        "Potable Probability",
                        f"{potable_probability:.2f}%"
                    )

                with prob_col2:
                    st.metric(
                        "Non-Potable Probability",
                        f"{non_potable_probability:.2f}%"
                    )

        except Exception as e:

            st.error("Prediction failed.")

            st.exception(e)

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Water Potability Prediction Using Machine Learning | "
    "Random Forest and SVM based classification project"
)

