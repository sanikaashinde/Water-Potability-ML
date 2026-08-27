from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.lib.units import inch
from pathlib import Path

root = Path(".")
out = root / "reports" / "final_report.pdf"
outputs = root / "outputs"

doc = SimpleDocTemplate(
    str(out),
    pagesize=A4,
    rightMargin=45,
    leftMargin=45,
    topMargin=45,
    bottomMargin=45
)

styles = getSampleStyleSheet()

title = ParagraphStyle(
    "TitleCustom",
    parent=styles["Title"],
    alignment=TA_CENTER,
    fontSize=20,
    leading=25,
    spaceAfter=20
)

heading = ParagraphStyle(
    "HeadingCustom",
    parent=styles["Heading2"],
    fontSize=14,
    leading=18,
    spaceBefore=12,
    spaceAfter=8
)

body = ParagraphStyle(
    "BodyCustom",
    parent=styles["BodyText"],
    fontSize=10,
    leading=15,
    spaceAfter=8
)

small = ParagraphStyle(
    "Small",
    parent=styles["BodyText"],
    fontSize=8,
    leading=11
)

story = []

story.append(Paragraph(
    "Water Potability Prediction Using Machine Learning",
    title
))

story.append(Paragraph(
    "<b>Final Project Report</b><br/>"
    "Machine Learning based Water Quality Classification System",
    ParagraphStyle("sub", parent=body, alignment=TA_CENTER)
))
story.append(Spacer(1, 20))

story.append(Paragraph("1. Abstract", heading))
story.append(Paragraph(
    "This project develops a machine learning based water quality classification "
    "system to predict whether a water sample is Potable (Safe to Drink) or "
    "Non-Potable (Not Safe to Drink). The project uses the Kaggle Water Potability "
    "Dataset containing 3276 samples and nine physicochemical water-quality "
    "features. Multiple classification algorithms were evaluated, including "
    "Random Forest, Support Vector Machine (SVM), Logistic Regression, Decision "
    "Tree, and K-Nearest Neighbors (KNN). A Hyperparameter-Tuned Random Forest "
    "was developed as the proposed model using GridSearchCV. An interactive "
    "Streamlit dashboard was also developed for dataset analysis, model "
    "evaluation, model comparison, and real-time prediction.",
    body
))

story.append(Paragraph("2. Problem Statement", heading))
story.append(Paragraph(
    "Determining whether water is suitable for drinking is an important water "
    "quality classification problem. Manual analysis can require laboratory "
    "testing and expert interpretation. This project investigates whether "
    "machine learning can classify water samples using physicochemical "
    "measurements and provide an interactive prediction system.",
    body
))

story.append(Paragraph("3. Objectives", heading))
for item in [
    "Analyze the Water Potability Dataset.",
    "Perform exploratory data analysis and preprocessing.",
    "Handle missing values appropriately.",
    "Split the dataset into training, validation, and testing sets.",
    "Train and compare multiple machine learning classification models.",
    "Implement Random Forest and SVM as compulsory models.",
    "Develop a Hyperparameter-Tuned Random Forest as the proposed model.",
    "Evaluate models using Accuracy, Precision, Recall, F1-Score and ROC-AUC.",
    "Perform confusion matrix and 5-fold cross-validation analysis.",
    "Develop an interactive Streamlit dashboard for prediction."
]:
    story.append(Paragraph("• " + item, body))

story.append(Paragraph("4. Dataset", heading))
dataset_data = [
    ["Property", "Value"],
    ["Dataset", "Kaggle Water Potability Dataset"],
    ["Total Samples", "3276"],
    ["Input Features", "9"],
    ["Total Columns", "10"],
    ["Target Column", "Potability"],
    ["Duplicate Rows", "0"],
    ["Classes", "2"]
]
t = Table(dataset_data, colWidths=[2.2*inch, 3.5*inch])
t.setStyle(TableStyle([
    ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
    ("GRID", (0,0), (-1,-1), 0.5, colors.grey),
    ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
    ("VALIGN", (0,0), (-1,-1), "TOP"),
    ("PADDING", (0,0), (-1,-1), 6)
]))
story.append(t)
story.append(Spacer(1, 10))

story.append(Paragraph("5. Input Features", heading))
story.append(Paragraph(
    "The nine input features are: pH, Hardness, Solids, Chloramines, Sulfate, "
    "Conductivity, Organic Carbon, Trihalomethanes, and Turbidity.",
    body
))
story.append(Paragraph(
    "<b>Target:</b> 0 = Non-Potable / Not Safe, 1 = Potable / Safe.",
    body
))

story.append(Paragraph("6. Data Preprocessing", heading))
story.append(Paragraph(
    "The preprocessing workflow includes dataset loading, duplicate checking, "
    "missing-value analysis, missing-value imputation, train-validation-test "
    "splitting, and feature scaling. A preprocessing pipeline was saved using "
    "joblib so that the same transformations can be applied consistently to "
    "new prediction data.",
    body
))

story.append(Paragraph("7. Machine Learning Models", heading))
models = [
    ["Model", "Purpose"],
    ["Random Forest", "Compulsory baseline model"],
    ["SVM", "Compulsory classification model"],
    ["Logistic Regression", "Comparison model"],
    ["Decision Tree", "Comparison model"],
    ["KNN", "Comparison model"],
    ["Tuned Random Forest", "Proposed improved model"]
]
t = Table(models, colWidths=[2.3*inch, 3.4*inch])
t.setStyle(TableStyle([
    ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
    ("GRID", (0,0), (-1,-1), 0.5, colors.grey),
    ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
    ("PADDING", (0,0), (-1,-1), 6)
]))
story.append(t)

story.append(Paragraph("8. Proposed Model", heading))
story.append(Paragraph(
    "The proposed model is a Hyperparameter-Tuned Random Forest. "
    "GridSearchCV was used to search for an effective combination of model "
    "hyperparameters.",
    body
))
params = [
    ["Parameter", "Selected Value"],
    ["n_estimators", "300"],
    ["max_depth", "10"],
    ["max_features", "sqrt"],
    ["min_samples_leaf", "1"],
    ["min_samples_split", "5"]
]
t = Table(params, colWidths=[2.5*inch, 3.2*inch])
t.setStyle(TableStyle([
    ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
    ("GRID", (0,0), (-1,-1), 0.5, colors.grey),
    ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
    ("PADDING", (0,0), (-1,-1), 6)
]))
story.append(t)

story.append(Paragraph("9. Validation Model Comparison", heading))
validation = [
    ["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"],
    ["Tuned Random Forest", "0.6517", "0.5714", "0.4188", "0.4834", "0.6594"],
    ["SVM", "0.6273", "0.5208", "0.5236", "0.5222", "0.6570"],
    ["Random Forest", "0.6436", "0.5588", "0.3979", "0.4648", "0.6505"],
    ["KNN", "0.6314", "0.5379", "0.3717", "0.4396", "0.6286"],
    ["Decision Tree", "0.6090", "0.4970", "0.4346", "0.4637", "0.5773"],
    ["Logistic Regression", "0.5214", "0.4141", "0.5550", "0.4743", "0.5331"]
]
t = Table(validation, colWidths=[1.65*inch, .72*inch, .72*inch, .72*inch, .72*inch, .72*inch])
t.setStyle(TableStyle([
    ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
    ("GRID", (0,0), (-1,-1), 0.4, colors.grey),
    ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
    ("FONTSIZE", (0,0), (-1,-1), 7),
    ("PADDING", (0,0), (-1,-1), 4)
]))
story.append(t)

story.append(Paragraph(
    "The Hyperparameter-Tuned Random Forest achieved the highest validation "
    "accuracy and precision among the compared models and was selected as "
    "the proposed model.",
    body
))

story.append(Paragraph("10. Cross-Validation Results", heading))
cv = [
    ["Model", "Mean ROC-AUC", "Std. Deviation"],
    ["SVM", "0.7037", "0.0140"],
    ["Random Forest", "0.6918", "0.0174"],
    ["Tuned Random Forest", "0.6892", "0.0213"],
    ["KNN", "0.6278", "0.0113"],
    ["Decision Tree", "0.5563", "0.0243"],
    ["Logistic Regression", "0.4855", "0.0181"]
]
t = Table(cv, colWidths=[2.8*inch, 1.5*inch, 1.5*inch])
t.setStyle(TableStyle([
    ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
    ("GRID", (0,0), (-1,-1), 0.4, colors.grey),
    ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
    ("PADDING", (0,0), (-1,-1), 5)
]))
story.append(t)

story.append(Paragraph(
    "SVM achieved the highest mean cross-validation ROC-AUC. However, model "
    "selection was based on overall validation performance across multiple "
    "metrics rather than a single metric.",
    body
))

story.append(PageBreak())

story.append(Paragraph("11. Final Test Results", heading))
final_metrics = [
    ["Metric", "Result"],
    ["Accuracy", "64.84%"],
    ["Precision", "55.76%"],
    ["Recall", "47.92%"],
    ["F1-Score", "51.54%"],
    ["ROC-AUC", "0.6758"]
]
t = Table(final_metrics, colWidths=[3*inch, 2.8*inch])
t.setStyle(TableStyle([
    ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
    ("GRID", (0,0), (-1,-1), 0.5, colors.grey),
    ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
    ("PADDING", (0,0), (-1,-1), 6)
]))
story.append(t)

story.append(Paragraph("12. Final Confusion Matrix", heading))
cm = [
    ["", "Predicted Non-Potable", "Predicted Potable"],
    ["Actual Non-Potable", "227", "73"],
    ["Actual Potable", "100", "92"]
]
t = Table(cm, colWidths=[2.0*inch, 1.8*inch, 1.8*inch])
t.setStyle(TableStyle([
    ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
    ("GRID", (0,0), (-1,-1), 0.5, colors.grey),
    ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
    ("PADDING", (0,0), (-1,-1), 6)
]))
story.append(t)

story.append(Paragraph(
    "The model correctly classified 227 non-potable samples and 92 potable "
    "samples. It incorrectly classified 73 non-potable samples as potable and "
    "100 potable samples as non-potable.",
    body
))

story.append(Paragraph("13. Comparison with Base Paper", heading))
story.append(Paragraph(
    "The base paper reports Random Forest and SVM as important classification "
    "models, with Random Forest performing better in its reported results. "
    "In this implementation, SVM can obtain stronger results on some evaluation "
    "measures. This difference is acceptable because machine learning results "
    "can vary with preprocessing, missing-value handling, data splitting, "
    "dataset distribution, hyperparameter settings, and evaluation methodology.",
    body
))

story.append(Paragraph(
    "SVM achieved the highest 5-fold cross-validation ROC-AUC of 0.7037. "
    "However, SVM was not labelled as the proposed model. The proposed model "
    "is the Hyperparameter-Tuned Random Forest, selected based on overall "
    "validation performance.",
    body
))

story.append(Paragraph("14. Streamlit Dashboard", heading))
story.append(Paragraph(
    "An interactive Streamlit dashboard was developed with five major sections: "
    "Dashboard, Dataset Summary, Final Model Evaluation, Model Comparison, "
    "and Water Potability Prediction.",
    body
))

for img_name, caption in [
    ("dashboard.JPG", "Streamlit Dashboard"),
    ("dataset_summary.JPG", "Dataset Summary"),
    ("model_evaluation.JPG", "Final Model Evaluation"),
    ("model_comparison.JPG", "Model Comparison"),
    ("prediction.JPG", "Water Potability Prediction")
]:
    img_path = root / "screenshots" / img_name
    if img_path.exists():
        story.append(Paragraph(caption, heading))
        im = Image(str(img_path))
        im._restrictSize(6.5*inch, 4.0*inch)
        story.append(im)
        story.append(Spacer(1, 8))

story.append(PageBreak())

story.append(Paragraph("15. Prediction System", heading))
story.append(Paragraph(
    "The prediction page accepts the nine physicochemical parameters: pH, "
    "Hardness, Solids, Chloramines, Sulfate, Conductivity, Organic Carbon, "
    "Trihalomethanes, and Turbidity. The trained proposed model and saved "
    "preprocessing pipeline are loaded from the models folder. The application "
    "returns a Potable or Non-Potable prediction together with potable and "
    "non-potable probabilities.",
    body
))

story.append(Paragraph("16. Project Structure", heading))
structure = """
Water-Potability-ML/<br/>
├── app.py<br/>
├── README.md<br/>
├── requirements.txt<br/>
├── data/<br/>
├── models/<br/>
│   ├── best_model.joblib<br/>
│   └── preprocessor.joblib<br/>
├── notebooks/<br/>
├── outputs/<br/>
├── screenshots/<br/>
└── src/<br/>
    ├── preprocessing.py<br/>
    ├── train_models.py<br/>
    ├── model_comparison.py<br/>
    └── evaluate_models.py
"""
story.append(Paragraph(structure, body))

story.append(Paragraph("17. Technologies Used", heading))
story.append(Paragraph(
    "Python, Pandas, NumPy, Scikit-learn, Joblib, Matplotlib, and Streamlit "
    "were used to develop the project.",
    body
))

story.append(Paragraph("18. Conclusion", heading))
story.append(Paragraph(
    "This project demonstrates an end-to-end machine learning workflow for "
    "water potability classification. Multiple classification models were "
    "implemented and compared using standard evaluation metrics. Random Forest "
    "and SVM were included as compulsory models, while a Hyperparameter-Tuned "
    "Random Forest was developed as the proposed model. The proposed model "
    "achieved 64.84% accuracy and 0.6758 ROC-AUC on the unseen test dataset. "
    "The Streamlit dashboard provides an interactive interface for dataset "
    "analysis, model evaluation, model comparison, and real-time water "
    "potability prediction.",
    body
))

story.append(Spacer(1, 20))
story.append(Paragraph(
    "<b>Final Model:</b> Hyperparameter-Tuned Random Forest<br/>"
    "<b>Final Test Accuracy:</b> 64.84%<br/>"
    "<b>Final Test ROC-AUC:</b> 0.6758",
    body
))

doc.build(story)

print(f"FINAL REPORT CREATED: {out}")
