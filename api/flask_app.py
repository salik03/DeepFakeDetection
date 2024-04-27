from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.applications import Xception
import base64
from io import BytesIO
from PIL import Image
from flask_cors import CORS
import requests
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

app = Flask(__name__)
CORS(app)

def load_deepfake_model(weights_url):
    # Download weights if not already downloaded
    weights_filename = 'xception_weights.h5'
    if not os.path.exists(weights_filename):
        response = requests.get(weights_url)
        with open(weights_filename, 'wb') as f:
            f.write(response.content)

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

model = load_deepfake_model('https://github.com/salik03/DeepFakeDetection/raw/main/xception_weights.h5')

def decode_and_preprocess_image(base64_image):
    image_bytes = base64.b64decode(base64_image)
    img = Image.open(BytesIO(image_bytes))
    img = img.resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    base64_image = data['image']
    image_array = decode_and_preprocess_image(base64_image)
    prediction = model.predict(image_array)
    probability = prediction[0][0]
    return jsonify({'probability': float(probability)})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
