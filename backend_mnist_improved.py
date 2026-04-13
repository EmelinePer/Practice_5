import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tensorflow as tf
import base64
import io
from PIL import Image, ImageOps

app = FastAPI(title="MNIST Digit Predictor API - Improved")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the trained model once at startup
try:
    model = tf.keras.models.load_model("mnist_model_improved.h5")
    print("✅ Model loaded successfully.")
except FileNotFoundError:
    print("❌ Model file not found! Run train_mnist_improved.py first.")
    model = None


class ImageData(BaseModel):
    # base64-encoded PNG image string (with or without data URI prefix)
    image: str


def preprocess_image(image_b64: str) -> np.ndarray:
    """
    Decode base64 image and preprocess to match MNIST format:
    - White digit on black background
    - Digit centered and fitted in 20x20, padded to 28x28 (MNIST style)
    """
    if "," in image_b64:
        image_b64 = image_b64.split(",")[1]

    image_bytes = base64.b64decode(image_b64)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGBA")

    # Flatten transparency onto black background (canvas background is black)
    background = Image.new("RGBA", image.size, (0, 0, 0, 255))
    background.paste(image, mask=image.split()[3])
    image = background.convert("L")  # grayscale

    # At this point: white stroke on black background = correct MNIST polarity
    arr = np.array(image).astype("float32")

    # Threshold to remove noise
    arr[arr < 50] = 0

    # Find bounding box of the drawn digit
    rows = np.any(arr > 0, axis=1)
    cols = np.any(arr > 0, axis=0)

    if not rows.any() or not cols.any():
        # Nothing drawn — return blank
        return np.zeros((1, 28, 28, 1), dtype="float32")

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    # Crop tightly around the digit
    cropped = arr[rmin:rmax+1, cmin:cmax+1]

    # Resize to 20x20 keeping aspect ratio (MNIST digits fit in ~20x20)
    h, w = cropped.shape
    if h > w:
        new_h, new_w = 20, max(1, int(20 * w / h))
    else:
        new_h, new_w = max(1, int(20 * h / w)), 20

    pil_crop = Image.fromarray(cropped).resize((new_w, new_h), Image.LANCZOS)
    resized = np.array(pil_crop)

    # Paste onto 28x28 black canvas, centered
    canvas = np.zeros((28, 28), dtype="float32")
    y_off = (28 - new_h) // 2
    x_off = (28 - new_w) // 2
    canvas[y_off:y_off+new_h, x_off:x_off+new_w] = resized

    # Normalize to [0, 1]
    canvas = canvas / 255.0

    return canvas.reshape(1, 28, 28, 1)


@app.get("/")
def root():
    return {"message": "MNIST Digit Predictor API (Improved) is running"}


@app.post("/predict")
def predict(data: ImageData):
    if model is None:
        return {"error": "Model not loaded"}
    
    arr = preprocess_image(data.image)
    predictions = model.predict(arr)[0]  # shape (10,)
    digit = int(np.argmax(predictions))
    confidence = float(predictions[digit])

    # Return top-3 predictions
    top3_idx = np.argsort(predictions)[::-1][:3]
    top3 = [
        {"digit": int(i), "probability": float(predictions[i])}
        for i in top3_idx
    ]

    return {
        "digit": digit,
        "confidence": confidence,
        "top3": top3,
    }