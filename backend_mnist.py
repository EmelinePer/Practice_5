from fastapi import FastAPI, UploadFile, File
import numpy as np
import cv2
import tensorflow as tf

app = FastAPI(title="MNIST Predictor API")
model = tf.keras.models.load_model('mnist_cnn.h5')

def center_and_preprocess(img_array):
    # Resize to 28x28
    img = cv2.resize(img_array, (28, 28), interpolation=cv2.INTER_AREA)
    # Normalize
    img = img / 255.0
    return img.reshape(1, 28, 28, 1)

@app.post("/predict")
async def predict_mnist(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    
    processed_img = improve_preprocess(img)
    pred = model.predict(processed_img)
    digit = int(np.argmax(pred))
    
    return {"digit": digit, "confidence": float(np.max(pred))}

def improve_preprocess(img_array):
    # 1. Threshold to get pure black and white
    _, thresh = cv2.threshold(img_array, 127, 255, cv2.THRESH_BINARY)
    
    # 2. Find contours to crop tightly around the digit
    coords = cv2.findNonZero(thresh)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        thresh = thresh[y:y+h, x:x+w]
    else:
        return np.zeros((1, 28, 28, 1))

    # 3. Resize to fit entirely in 20x20, preserving aspect ratio!
    scale = 20.0 / max(thresh.shape[0], thresh.shape[1])
    new_w = max(int(thresh.shape[1] * scale), 1)
    new_h = max(int(thresh.shape[0] * scale), 1)
    res = cv2.resize(thresh, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # 4. Center in a 28x28 square
    final_img = np.zeros((28, 28), dtype=np.uint8)
    y_offset = (28 - new_h) // 2
    x_offset = (28 - new_w) // 2
    final_img[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = res
    
    return final_img.reshape(1, 28, 28, 1) / 255.0