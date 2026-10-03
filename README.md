# 🏥 AI Health Risk Prediction System

## 📌 Project Overview

The **AI Health Risk Prediction System** is a Machine Learning-based web application that predicts whether a person's health information falls into a **lower estimated risk** or **higher estimated risk** category.

The system takes health-related input values, processes them, and uses a trained Machine Learning model to generate an estimated prediction.

> ⚠️ **Disclaimer:** This project is developed for educational and research purposes only. It is not a medical diagnosis system and should not be used for making medical decisions.

---

## 🎯 Problem Statement

Health-related datasets contain multiple factors that can be difficult to analyze manually.

This project demonstrates how Machine Learning can be used to analyze historical health data and provide a simple estimated risk category based on the information provided by the user.

---

## 💡 How the System Works

The project follows this basic workflow:

**User Input → Data Processing → Machine Learning Model → Prediction → Result**

1. The user enters health-related information.
2. The input values are converted into the format required by the model.
3. The trained Machine Learning model processes the information.
4. The model predicts one of two categories:

   * **Lower Estimated Risk**
   * **Higher Estimated Risk**
5. The result is displayed through the web application.

---

## 🧠 Machine Learning Model

The project uses **Logistic Regression**, a supervised Machine Learning algorithm commonly used for classification problems.

In this project, the model performs binary classification:

* `0` → Lower estimated risk category
* `1` → Higher estimated risk category

The model was trained using historical heart-disease data.

---

## 📊 Dataset

The project uses the **Cleveland Heart Disease dataset**.

The dataset contains health-related features such as:

* Age
* Sex
* Chest pain type
* Resting blood pressure
* Cholesterol
* Fasting blood sugar
* Resting ECG
* Maximum heart rate
* Exercise-induced chest pain
* Oldpeak
* Slope
* CA
* Thal

The original target values were converted into two categories for binary classification.

---

## 🧹 Data Preprocessing

Before training the model, the data was prepared using the following steps:

* Missing values were identified.
* Missing values in `ca` and `thal` were handled using median values.
* The original target values were converted into binary classes.
* The dataset was separated into:

  * **X** → input features
  * **y** → target
* The data was divided into training and testing sets using an **80/20 split**.

---

## 📈 Model Evaluation

The model was evaluated using:

### Accuracy

Measures how many predictions were classified correctly.

### Confusion Matrix

Shows correct and incorrect predictions for both classes.

### Classification Report

Provides:

* Precision
* Recall
* F1-score
* Support

A separate **Model Performance** page is included in the application to display these evaluation results.

> The reported performance represents results on the project's test dataset and should not be interpreted as clinical accuracy.

---

## 🖥️ Web Application

The user interface was developed using **Streamlit**.

The application allows the user to enter health-related information and receive the model's estimated risk category.

The application also includes a separate model-performance section showing evaluation metrics and the confusion matrix.

---

## 🛠️ Technologies Used

* **Python**
* **Pandas**
* **Scikit-learn**
* **Joblib**
* **Matplotlib**
* **Seaborn**
* **Streamlit**
* **Git & GitHub**
* **VS Code**

---

## 📁 Project Structure

```text
AI-Health-Risk-Prediction/
│
├── app.py
├── load_data.py
├── model_performance.py
├── heart_disease.data
├── health_risk_model.pkl
├── requirements.txt
├── README.md
└── .gitignore
```

### File Description

**`app.py`**
Main Streamlit application.

**`load_data.py`**
Loads the dataset, preprocesses the data, trains the Logistic Regression model, evaluates it, and saves the trained model.

**`model_performance.py`**
Displays model accuracy, confusion matrix, and classification report.

**`heart_disease.data`**
Historical dataset used for training and testing.

**`health_risk_model.pkl`**
Saved trained Machine Learning model.

**`requirements.txt`**
Contains the Python libraries required to run the project.

**`.gitignore`**
Prevents unnecessary files such as the virtual environment and Python cache files from being uploaded to GitHub.

---

## ▶️ How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/nithinj-ds/AI-Health-Risk-Prediction.git
```

### 2. Open the project folder

```bash
cd AI-Health-Risk-Prediction
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

### 4. Activate the virtual environment

**Windows PowerShell:**

```powershell
.\venv\Scripts\Activate.ps1
```

### 5. Install the required libraries

```bash
pip install -r requirements.txt
```

### 6. Run the Streamlit application

```bash
python -m streamlit run app.py
```

The application will open in your browser.

---

## 🔐 Important Note

This system is **not a replacement for a doctor, medical examination, or professional healthcare advice**.

The prediction is based on a historical dataset and a Machine Learning model developed for educational purposes.

---

## 👨‍💻 Project

**AI Health Risk Prediction System**

Developed as an AI & Data Science academic project.
