import streamlit as st
import requests
import os


def resolve_backend_url() -> str:
    try:
        secret_url = st.secrets.get("BACKEND_URL")
    except Exception:
        secret_url = None
    env_url = os.getenv("BACKEND_URL")
    return (secret_url or env_url or "http://127.0.0.1:8000").rstrip("/")


if "backend_url" not in st.session_state:
    st.session_state.backend_url = resolve_backend_url()

st.title("🛸 UFO Appearance Prediction! 👽")
st.write("Enter coordinates and duration to predict the country.")

backend_url = st.sidebar.text_input(
    "Backend URL",
    value=st.session_state.backend_url,
    help="For Streamlit Cloud, use your public FastAPI URL.",
)
BACKEND_URL = backend_url.rstrip("/")
st.session_state.backend_url = BACKEND_URL

seconds = st.number_input("Seconds", min_value=0, max_value=60, value=10)
latitude = st.number_input("Latitude", value=50.0)
longitude = st.number_input("Longitude", value=-12.0)

if st.button("Predict Country"):
    payload = {"seconds": seconds, "latitude": latitude, "longitude": longitude}
    try:
        res = requests.post(f"{BACKEND_URL}/predict", json=payload, timeout=10)
    except requests.exceptions.ConnectionError:
        st.error(
            f"Cannot connect to backend at {BACKEND_URL}. "
            "Set a public API URL if running on Streamlit Cloud."
        )
        st.stop()
    if res.status_code == 200:
        st.success(f"Likely country: {res.json()['country']}")
    else:
        st.error("Error connecting to backend API")