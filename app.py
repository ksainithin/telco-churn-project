import joblib
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide",
)


@st.cache_resource
def load_model():
    return joblib.load(
        "models/churn_xgboost_pipeline.joblib"
    )


model = load_model()

st.title("Customer Churn Risk Predictor")

st.write(
    "Enter a customer's information to estimate "
    "whether the customer is at risk of leaving."
)

with st.form("customer_form"):
    st.subheader("Customer information")

    column_1, column_2, column_3 = st.columns(3)

    with column_1:
        gender = st.selectbox(
            "Gender",
            ["Female", "Male"],
        )

        senior_citizen = st.selectbox(
            "Senior citizen",
            [0, 1],
            format_func=lambda value: "Yes" if value == 1 else "No",
        )

        partner = st.selectbox(
            "Has a partner",
            ["No", "Yes"],
        )

        dependents = st.selectbox(
            "Has dependents",
            ["No", "Yes"],
        )

        tenure = st.slider(
            "Tenure in months",
            min_value=0,
            max_value=72,
            value=12,
        )

    with column_2:
        phone_service = st.selectbox(
            "Phone service",
            ["No", "Yes"],
        )

        multiple_lines = st.selectbox(
            "Multiple phone lines",
            ["No", "Yes", "No phone service"],
        )

        internet_service = st.selectbox(
            "Internet service",
            ["DSL", "Fiber optic", "No"],
        )

        online_security = st.selectbox(
            "Online security",
            ["No", "Yes", "No internet service"],
        )

        online_backup = st.selectbox(
            "Online backup",
            ["No", "Yes", "No internet service"],
        )

        device_protection = st.selectbox(
            "Device protection",
            ["No", "Yes", "No internet service"],
        )

    with column_3:
        tech_support = st.selectbox(
            "Technical support",
            ["No", "Yes", "No internet service"],
        )

        streaming_tv = st.selectbox(
            "Streaming TV",
            ["No", "Yes", "No internet service"],
        )

        streaming_movies = st.selectbox(
            "Streaming movies",
            ["No", "Yes", "No internet service"],
        )

        contract = st.selectbox(
            "Contract",
            ["Month-to-month", "One year", "Two year"],
        )

        paperless_billing = st.selectbox(
            "Paperless billing",
            ["No", "Yes"],
        )

    st.subheader("Billing information")

    billing_column_1, billing_column_2, billing_column_3 = st.columns(3)

    with billing_column_1:
        payment_method = st.selectbox(
            "Payment method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
        )

    with billing_column_2:
        monthly_charges = st.number_input(
            "Monthly charges ($)",
            min_value=0.0,
            max_value=150.0,
            value=70.0,
            step=1.0,
        )

    with billing_column_3:
        total_charges = st.number_input(
            "Total charges ($)",
            min_value=0.0,
            max_value=10000.0,
            value=840.0,
            step=10.0,
        )

    submitted = st.form_submit_button(
        "Predict churn risk",
        type="primary",
    )


if submitted:
    customer_data = pd.DataFrame({
        "SeniorCitizen": [senior_citizen],
        "tenure": [tenure],
        "MonthlyCharges": [monthly_charges],
        "TotalCharges": [total_charges],
        "gender": [gender],
        "Partner": [partner],
        "Dependents": [dependents],
        "PhoneService": [phone_service],
        "MultipleLines": [multiple_lines],
        "InternetService": [internet_service],
        "OnlineSecurity": [online_security],
        "OnlineBackup": [online_backup],
        "DeviceProtection": [device_protection],
        "TechSupport": [tech_support],
        "StreamingTV": [streaming_tv],
        "StreamingMovies": [streaming_movies],
        "Contract": [contract],
        "PaperlessBilling": [paperless_billing],
        "PaymentMethod": [payment_method],
    })

    churn_score = model.predict_proba(customer_data)[0, 1]
    prediction = model.predict(customer_data)[0]

    st.divider()
    st.subheader("Prediction result")

    st.metric(
        "Model churn score",
        f"{churn_score:.1%}",
    )

    if prediction == 1:
        st.error(
            "This customer is predicted to be at risk of churn."
        )
    else:
        st.success(
            "This customer is predicted to have lower churn risk."
        )

    st.caption(
        "This score is a model estimate for learning purposes, "
        "not a guaranteed probability."
    )