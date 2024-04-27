import cv2
import os
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import keras
from keras import layers
from sklearn import metrics
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import BatchNormalization, ReLU, Conv2D, concatenate, ZeroPadding2D, AvgPool2D, MaxPool2D,GlobalAveragePooling2D, Dense, Input, Flatten
from tensorflow.keras.models import Model, Sequential
from keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications import Xception

def limit_data(data_dir, n=100):
    a = []
    for i in os.listdir(data_dir):
        if i == '.DS_Store':
            continue
        for k, j in enumerate(os.listdir(os.path.join(data_dir, i))):
            if k > n:
                continue
            a.append((os.path.join(data_dir, i, j), i))
            print(i,j)
    return pd.DataFrame(a, columns=['filename', 'class'])
base_path = 'real-vs-fake/'
image_gen = ImageDataGenerator(rescale=1./255.)
batch_size = 32
print("Organised")
'''
train_df = limit_data(base_path+'train',50000)
valid_df = limit_data(base_path+'valid',10000)
test_df = limit_data(base_path+'test',10000)
'''
train_flow = image_gen.flow_from_directory(
    directory="real-vs-fake/train",
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode="binary")
valid_flow = image_gen.flow_from_directory(
    directory="real-vs-fake/valid",
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode="binary")
test_flow = image_gen.flow_from_directory(
    directory="real-vs-fake/test",
    target_size=(224, 224),
    batch_size=1,
    shuffle=False,
    class_mode="binary")

# Load Xception model without the top (classification) layers
xception_model = Xception(input_shape=(224, 224, 3), weights='imagenet', include_top=False)

# Freeze the pretrained layers
for layer in xception_model.layers[:-4]:
    layer.trainable = False

# Add new classification layers
output = xception_model.layers[-1].output
output = GlobalAveragePooling2D()(output)
output = Dense(256, activation='relu')(output)
output = Dense(1, activation='sigmoid')(output)

# Create a new model
model = Model(inputs=xception_model.input, outputs=output)

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Print a summary of the model architecture
model.summary()

# Train the model
# model.fit_generator(train_flow, steps_per_epoch=20, epochs=100, validation_data=valid_flow, validation_steps=100, verbose=1)
model.fit_generator(train_flow, epochs=20, validation_data=valid_flow, validation_steps=100, verbose=1)

# Save the model weights
model.save_weights('xception_weights.h5')

# Evaluate the model on the test set
evaluation = model.evaluate_generator(test_flow)
print("Test Loss:", evaluation[0])
print("Test Accuracy:", evaluation[1])
print(evaluation)