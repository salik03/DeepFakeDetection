from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.applications import Xception
import base64
from io import BytesIO
from PIL import Image
from fastapi.middleware.cors import CORSMiddleware
import requests
import os
import uvicorn

app = FastAPI()

# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ImageData(BaseModel):
    image: str

def load_deepfake_model(weights_url):
    # Download weights if not already downloaded
    weights_filename = 'xception_weights.h5'
    if not os.path.exists(weights_filename):
        print("Downloading model weights...")
        response = requests.get(weights_url)
        with open(weights_filename, 'wb') as f:
            f.write(response.content)

    # Check the file size to confirm it's not empty or incomplete
    file_size = os.path.getsize(weights_filename)
    print(f"File Size: {file_size} bytes")  # Add this to check file size

    if file_size < 1000:  # Example threshold, adjust based on expected size
        print("Downloaded file is too small. There may have been a download error.")
        return None

    xception_model = Xception(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
    for layer in xception_model.layers[:-4]:
        layer.trainable = False

    output = xception_model.output
    output = GlobalAveragePooling2D()(output)
    output = Dense(256, activation='relu')(output)
    output = Dense(1, activation='sigmoid')(output)
    model = Model(inputs=xception_model.input, outputs=output)
    model.load_weights(weights_filename)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


model = load_deepfake_model('https://github.com/salik03/DeepFakeDetection/raw/main/api/xception_weights.h5')

def decode_and_preprocess_image(base64_image):
    image_bytes = base64.b64decode(base64_image)
    img = Image.open(BytesIO(image_bytes))
    img = img.resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.post('/predict/')
def predict(image_data: ImageData):
    image_array = decode_and_preprocess_image(image_data.image)
    prediction = model.predict(image_array)
    probability = prediction[0][0]
    return JSONResponse(content={'probability': float(probability)})

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=5000)
