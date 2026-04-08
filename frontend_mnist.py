import streamlit as st
from streamlit_drawable_canvas import st_canvas
import requests
import cv2

st.title("✍️ MNIST Digit Recognizer")
st.write("Draw a digit below!")

# Create a canvas component
canvas_result = st_canvas(
    fill_color="black", 
    stroke_width=20,     # Thicker stroke matches MNIST dataset better
    stroke_color="white",
    background_color="black",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas",
)

if st.button("Predict Digit"):
    if canvas_result.image_data is not None:
        img = canvas_result.image_data
        # Convert RGBA to Grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
        
        # Encode as PNG to send via HTTP
        _, encoded_img = cv2.imencode('.png', gray)
        
        res = requests.post("http://127.0.0.1:8001/predict", files={"file": encoded_img.tobytes()})
        if res.status_code == 200:
            data = res.json()
            st.success(f"Predicted Digit: {data['digit']} (Confidence: {data['confidence']:.2f})")