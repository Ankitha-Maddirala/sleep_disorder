import streamlit as st
import pickle


# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)


# Load scaler
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)


# Load occupation encoder
with open("occupation_encoder.pkl", "rb") as f:
    occupation_encoder = pickle.load(f)


st.title("Sleep Disorder Classification")

st.write(
    "Enter the sleep, health and lifestyle details "
    "to predict the sleep disorder category."
)


# User Inputs

gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)


age = st.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=30
)


occupation = st.selectbox(
    "Occupation",
    occupation_encoder.classes_
)


sleep_duration = st.number_input(
    "Sleep Duration (hours)",
    min_value=0.0,
    max_value=24.0,
    value=7.0
)


quality_of_sleep = st.number_input(
    "Quality of Sleep",
    min_value=1,
    max_value=10,
    value=7
)


physical_activity = st.number_input(
    "Physical Activity Level",
    min_value=0,
    max_value=300,
    value=60
)


stress_level = st.number_input(
    "Stress Level",
    min_value=1,
    max_value=10,
    value=5
)


bmi_category = st.selectbox(
    "BMI Category",
    ["Normal", "Normal Weight", "Overweight", "Obese"]
)


systolic = st.number_input(
    "Systolic Blood Pressure",
    min_value=80,
    max_value=250,
    value=120
)


diastolic = st.number_input(
    "Diastolic Blood Pressure",
    min_value=40,
    max_value=150,
    value=80
)


heart_rate = st.number_input(
    "Heart Rate",
    min_value=40,
    max_value=150,
    value=75
)


daily_steps = st.number_input(
    "Daily Steps",
    min_value=0,
    max_value=30000,
    value=6000
)


if st.button("Predict Sleep Disorder"):

    # Convert values to numbers

    gender_value = 0 if gender == "Male" else 1

    bmi_value = {
        "Normal": 0,
        "Normal Weight": 0,
        "Overweight": 1,
        "Obese": 2
    }[bmi_category]

    occupation_value = occupation_encoder.transform(
        [occupation]
    )[0]


    # Create input data

    input_data = [[
        gender_value,
        age,
        occupation_value,
        sleep_duration,
        quality_of_sleep,
        physical_activity,
        stress_level,
        bmi_value,
        systolic,
        diastolic,
        heart_rate,
        daily_steps
    ]]


    # Scale input

    input_scaled = scaler.transform(input_data)


    # Prediction

    prediction = model.predict(input_scaled)


    st.success(
        "Predicted Sleep Disorder: " + prediction[0]
    )