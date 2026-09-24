import streamlit as st
import joblib


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="AI Health Risk Prediction",
    page_icon="🏥",
    layout="wide"
)


# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown(
    """
    <style>

    .main-title {
        text-align: center;
        font-size: 40px;
        font-weight: bold;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        margin-bottom: 25px;
    }

    .result-box {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        margin-top: 15px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

model = joblib.load("health_risk_model.pkl")


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.header("🏥 About the System")

    st.write(
        "This application uses a Machine Learning model "
        "to estimate health risk based on the information entered."
    )

    st.divider()

    st.subheader("🧠 Machine Learning")

    st.write("Model used: Logistic Regression")

    st.write(
        "The model was trained using historical heart disease data."
    )

    st.divider()

    st.subheader("📊 System Process")

    st.write("1. Enter patient information")
    st.write("2. Enter available test results")
    st.write("3. AI model processes the information")
    st.write("4. Estimated risk category is displayed")

    st.divider()

    st.subheader("⚠️ Important")

    st.write(
        "This application is created for educational purposes "
        "and should not be used as a medical diagnosis."
    )


# --------------------------------------------------
# MAIN TITLE
# --------------------------------------------------

st.markdown(
    '<div class="main-title">🏥 AI Health Risk Prediction System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Machine Learning based health risk estimation</div>',
    unsafe_allow_html=True
)

st.caption(
    "⚠️ Educational project only. This prediction is not a medical diagnosis."
)


# --------------------------------------------------
# PATIENT INFORMATION
# --------------------------------------------------

st.divider()

st.header("👤 Patient Information")

col1, col2 = st.columns(2)


with col1:

    age = st.number_input(
        "Enter your age",
        min_value=1,
        max_value=120,
        value=30
    )

    cp = st.selectbox(
        "What type of chest pain is mentioned in the medical report?",
        [
            "Typical angina",
            "Atypical angina",
            "Non-anginal pain",
            "Asymptomatic"
        ],
        format_func=lambda x: {
            "Typical angina":
                "Typical angina — chest pressure/pain, often during activity",

            "Atypical angina":
                "Atypical angina — chest pain with some unusual symptoms",

            "Non-anginal pain":
                "Non-anginal pain — chest pain that does not match typical heart pain",

            "Asymptomatic":
                "Asymptomatic — no chest pain"
        }[x]
    )

    chol = st.number_input(
        "Enter cholesterol level",
        min_value=100,
        max_value=600,
        value=200
    )

    exang = st.selectbox(
        "Do you experience exercise-induced chest pain?",
        ["No", "Yes"]
    )


with col2:

    sex = st.selectbox(
        "Select sex",
        ["Female", "Male"]
    )

    trestbps = st.number_input(
        "Enter resting blood pressure",
        min_value=50,
        max_value=250,
        value=120
    )

    fbs = st.selectbox(
        "Is fasting blood sugar greater than 120 mg/dl?",
        ["No", "Yes"]
    )


# --------------------------------------------------
# MEDICAL TEST RESULTS
# --------------------------------------------------

st.divider()

st.header("🩺 Medical Test Results")

st.info(
    "For the following fields, enter the values shown in the "
    "patient's medical test report."
)

col3, col4 = st.columns(2)


with col3:

    restecg = st.selectbox(
        "What does your ECG report say?",
        [
            "Normal",
            "ST-T wave abnormality",
            "Left ventricular hypertrophy"
        ]
    )

    thalach = st.number_input(
        "Enter your highest heart rate during exercise",
        min_value=50,
        max_value=250,
        value=150
    )

    oldpeak = st.number_input(
        "Enter the ECG stress measurement from your report",
        min_value=0.0,
        max_value=10.0,
        value=0.0,
        step=0.1
    )


with col4:

    slope = st.selectbox(
        "Select the ECG pattern after exercise",
        [
            "Upsloping",
            "Flat",
            "Downsloping"
        ]
    )

    ca = st.number_input(
        "Enter the heart blood vessel test result",
        min_value=0,
        max_value=3,
        value=0,
        step=1
    )

    thal = st.selectbox(
        "Select the result shown in your blood-flow test report",
        [
            "Normal",
            "Fixed defect",
            "Reversible defect"
        ]
    )


# --------------------------------------------------
# CONVERT USER INPUTS TO MODEL VALUES
# --------------------------------------------------

sex_value = 1 if sex == "Male" else 0


cp_value = {
    "Typical angina": 1,
    "Atypical angina": 2,
    "Non-anginal pain": 3,
    "Asymptomatic": 4
}[cp]


fbs_value = 1 if fbs == "Yes" else 0


restecg_value = {
    "Normal": 0,
    "ST-T wave abnormality": 1,
    "Left ventricular hypertrophy": 2
}[restecg]


exang_value = 1 if exang == "Yes" else 0


slope_value = {
    "Upsloping": 1,
    "Flat": 2,
    "Downsloping": 3
}[slope]


thal_value = {
    "Normal": 3,
    "Fixed defect": 6,
    "Reversible defect": 7
}[thal]


# --------------------------------------------------
# PREPARE INPUT DATA
# --------------------------------------------------

input_data = [[
    age,
    sex_value,
    cp_value,
    trestbps,
    chol,
    fbs_value,
    restecg_value,
    thalach,
    exang_value,
    oldpeak,
    slope_value,
    ca,
    thal_value
]]


# --------------------------------------------------
# PREDICTION SECTION
# --------------------------------------------------

st.divider()

st.header("🔍 Generate Prediction")

st.write(
    "Review the entered information and click the button below "
    "to generate the AI-based estimated result."
)

predict_button = st.button(
    "🔍 Predict Health Risk",
    use_container_width=True
)


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

if predict_button:

    prediction = model.predict(input_data)

    st.divider()

    st.header("📊 Prediction Result")

    if prediction[0] == 1:

        st.warning("⚠️ Higher estimated risk detected")

        st.info(
            "The AI model detected a higher estimated risk category "
            "based on the information entered. This result is not a "
            "medical diagnosis. Please consult a qualified healthcare "
            "professional for medical advice."
        )

    else:

        st.success("✅ Lower estimated risk detected")

        st.info(
            "The AI model detected a lower estimated risk category "
            "based on the information entered. This result is not a "
            "medical diagnosis."
        )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "🏥 AI Health Risk Prediction System | "
    "Educational Machine Learning Project"
)