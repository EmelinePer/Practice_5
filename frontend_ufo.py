import streamlit as st
import requests

st.title("🛸 UFO Appearance Prediction! 👽")
st.write("Enter coordinates and duration to predict the country.")

seconds = st.number_input("Seconds", min_value=0, max_value=60, value=10)
latitude = st.number_input("Latitude", value=50.0)
longitude = st.number_input("Longitude", value=-12.0)

if st.button("Predict Country"):
    payload = {"seconds": seconds, "latitude": latitude, "longitude": longitude}
    res = requests.post("http://127.0.0.1:8000/predict", json=payload)
    if res.status_code == 200:
        st.success(f"Likely country: {res.json()['country']}")
    else:
        st.error("Error connecting to backend API")