import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Model Performance",
    page_icon="📊",
    layout="wide"
)


# --------------------------------------------------
# PAGE TITLE
# --------------------------------------------------

st.title("📊 Model Performance")

st.write(
    "This page shows how well the Machine Learning model "
    "performed when tested on historical data."
)

st.caption(
    "These results describe model performance on the test dataset "
    "and do not represent clinical accuracy."
)


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

columns = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "target"
]

data = pd.read_csv(
    "heart_disease.data",
    header=None,
    names=columns,
    na_values="?"
)


# --------------------------------------------------
# HANDLE MISSING VALUES
# --------------------------------------------------

data["ca"] = data["ca"].fillna(data["ca"].median())

data["thal"] = data["thal"].fillna(data["thal"].median())


# --------------------------------------------------
# CONVERT TARGET
# --------------------------------------------------

data["target"] = (data["target"] > 0).astype(int)


# --------------------------------------------------
# SEPARATE INPUT AND TARGET
# --------------------------------------------------

X = data.drop("target", axis=1)

y = data["target"]


# --------------------------------------------------
# SPLIT DATA
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# --------------------------------------------------
# TRAIN MODEL
# --------------------------------------------------

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)


# --------------------------------------------------
# MAKE PREDICTIONS
# --------------------------------------------------

predictions = model.predict(X_test)


# --------------------------------------------------
# ACCURACY
# --------------------------------------------------

accuracy = accuracy_score(
    y_test,
    predictions
)


# --------------------------------------------------
# DISPLAY ACCURACY
# --------------------------------------------------

st.subheader("🎯 Model Accuracy")

st.metric(
    "Accuracy",
    f"{accuracy * 100:.2f}%"
)


st.write(
    "Accuracy shows the percentage of test cases that "
    "the model classified correctly."
)


# --------------------------------------------------
# CONFUSION MATRIX
# --------------------------------------------------

st.subheader("🔢 Confusion Matrix")

cm = confusion_matrix(
    y_test,
    predictions
)


cm_df = pd.DataFrame(
    cm,
    index=["Actual 0", "Actual 1"],
    columns=["Predicted 0", "Predicted 1"]
)


st.dataframe(
    cm_df,
    use_container_width=True
)
# --------------------------------------------------
# CONFUSION MATRIX VISUALIZATION
# --------------------------------------------------

fig, ax = plt.subplots(figsize=(6, 4))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Predicted 0", "Predicted 1"],
    yticklabels=["Actual 0", "Actual 1"],
    ax=ax
)

ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("Confusion Matrix")

st.pyplot(fig)

st.write(
    "The confusion matrix shows the number of correct "
    "and incorrect predictions made by the model."
)


# --------------------------------------------------
# CLASSIFICATION REPORT
# --------------------------------------------------

st.subheader("📋 Classification Report")

report = classification_report(
    y_test,
    predictions,
    output_dict=True
)


report_df = pd.DataFrame(report).transpose()


st.dataframe(
    report_df.round(2),
    use_container_width=True
)


# --------------------------------------------------
# SIMPLE EXPLANATION
# --------------------------------------------------

st.subheader("🧠 What do these metrics mean?")

st.write(
    "**Precision:** How many predicted positive cases were actually positive."
)

st.write(
    "**Recall:** How many actual positive cases the model successfully detected."
)

st.write(
    "**F1-score:** A combined measure of precision and recall."
)

st.write(
    "**Support:** Number of test samples belonging to each class."
)