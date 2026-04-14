import streamlit as st
import requests
import base64
import numpy as np
from PIL import Image
import io
import os

try:
    from streamlit_drawable_canvas import st_canvas
except ModuleNotFoundError:
    st.set_page_config(
        page_title="MNIST Digit Recognizer",
        page_icon="✏️",
        layout="centered"
    )
    st.error(
        "The dependency 'streamlit-drawable-canvas' is missing in this deployment. "
        "Add it to requirements.txt and redeploy."
    )
    st.code("pip install streamlit-drawable-canvas>=0.9.3", language="bash")
    st.stop()

def resolve_backend_url() -> str:
    # Priority: Streamlit secrets > env var > local default
    secret_url = None
    try:
        secret_url = st.secrets.get("BACKEND_URL")
    except Exception:
        secret_url = None

    env_url = os.getenv("BACKEND_URL")
    url = secret_url or env_url or "http://127.0.0.1:8000"
    return url.rstrip("/")


if "backend_url" not in st.session_state:
    st.session_state.backend_url = resolve_backend_url()

st.set_page_config(
    page_title="MNIST Digit Recognizer", 
    page_icon="✏️", 
    layout="centered"
)

st.title("✏️ MNIST Handwritten Digit Recognizer")
st.markdown(
    "Draw a digit (0–9) in the canvas below, then click **Predict**."
)

# --- Sidebar settings ---
st.sidebar.header("🎨 Canvas Settings")
stroke_width = st.sidebar.slider("Pen width", 5, 50, 20)
stroke_color = st.sidebar.color_picker("Pen color", "#FFFFFF")
bg_color = st.sidebar.color_picker("Background color", "#000000")

st.sidebar.markdown("---")
backend_url = st.sidebar.text_input(
    "Backend URL",
    value=st.session_state.backend_url,
    help="For Streamlit Cloud, set your public FastAPI endpoint (e.g. https://my-api.onrender.com).",
)
BACKEND_URL = backend_url.rstrip("/")
st.session_state.backend_url = BACKEND_URL

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Model Info")
st.sidebar.info("""
**Architecture :**
- 2 Conv Blocks (32+64 filters)
- Batch Normalization
- Dropout (25% & 50%)
- Data Augmentation

**Expected Accuracy:** 99%+
""")

# --- Canvas ---
canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",
    stroke_width=stroke_width,
    stroke_color=stroke_color,
    background_color=bg_color,
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

col1, col2 = st.columns(2)

with col1:
    predict_btn = st.button("🔍 Predict", use_container_width=True, key="predict")

with col2:
    st.info("Use canvas toolbar (↺) to clear", icon="ℹ️")

# --- Prediction ---
if predict_btn:
    if canvas_result.image_data is None:
        st.warning("⚠️ Please draw something first!")
    else:
        # Convert numpy RGBA array → PNG → base64
        img_array = canvas_result.image_data.astype(np.uint8)
        pil_image = Image.fromarray(img_array, mode="RGBA")

        buffered = io.BytesIO()
        pil_image.save(buffered, format="PNG")
        img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        with st.spinner("🔮 Predicting..."):
            try:
                response = requests.post(
                    f"{BACKEND_URL}/predict",
                    json={"image": img_b64},
                    timeout=10,
                )
                response.raise_for_status()
                result = response.json()

                if "error" in result:
                    st.error(f"❌ Backend Error: {result['error']}")
                else:
                    digit = result["digit"]
                    confidence = result["confidence"]
                    top3 = result["top3"]

                    # Display main prediction
                    st.success(f"## 🎯 Predicted digit: **{digit}**")
                    st.metric("Confidence", f"{confidence * 100:.1f}%")

                    # Display top 3 predictions
                    st.subheader("📊 Top 3 predictions")
                    for item in top3:
                        col_digit, col_prob = st.columns([1, 4])
                        with col_digit:
                            st.write(f"**{item['digit']}**")
                        with col_prob:
                            st.progress(
                                item["probability"],
                                text=f"{item['probability']*100:.1f}%"
                            )

            except requests.exceptions.ConnectionError:
                st.error(
                    f"❌ Cannot connect to backend at {BACKEND_URL}\n\n"
                    "If you are on Streamlit Cloud, 127.0.0.1 will not work unless backend runs in the same container.\n\n"
                    "Set a public backend URL in the sidebar or via secret/env BACKEND_URL.\n\n"
                    "Local command:\n"
                    "uvicorn backend_mnist_improved:app --host 127.0.0.1 --port 8000 --reload"
                )
            except Exception as e:
                st.error(f"❌ Error: {e}")

# --- Footer ---
st.markdown("---")
st.caption("Backend: FastAPI + Uvicorn | Model: Advanced CNN with BatchNorm + Data Augmentation")