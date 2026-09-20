import streamlit as st
import pickle

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Load scaler
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)


st.title("Sleep Disorder Classification")

st.write("Enter the following details to predict the sleep disorder category.")


# User inputs

age = st.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=30
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
    "Physical Activity Level (minutes/day)",
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

    input_data = [[
        age,
        sleep_duration,
        quality_of_sleep,
        physical_activity,
        stress_level,
        heart_rate,
        daily_steps
    ]]

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)

    st.success("Predicted Sleep Disorder: " + prediction[0])