import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Activation
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
import cv2
from tqdm import tqdm

# Define paths
hr_dir = "DIV2K/DIV2K_train_HR/"
lr_dir = "DIV2K/DIV2K_train_LR/"

# Load images and preprocess
def load_images(hr_dir, lr_dir, size=(128, 128)):
    hr_images, lr_images = [], []
    for img_name in tqdm(os.listdir(hr_dir)[:500]):  # Load 500 images for training
        hr_img_path = os.path.join(hr_dir, img_name)
        lr_img_path = os.path.join(lr_dir, img_name)

        hr_img = cv2.imread(hr_img_path)
        lr_img = cv2.imread(lr_img_path)

        if hr_img is not None and lr_img is not None:
            hr_img = cv2.resize(hr_img, size)
            lr_img = cv2.resize(lr_img, size)
            hr_images.append(hr_img / 255.0)
            lr_images.append(lr_img / 255.0)
    
    return np.array(lr_images), np.array(hr_images)

# Load dataset
lr_images, hr_images = load_images(hr_dir, lr_dir)

# Split into training and validation sets
train_lr, val_lr, train_hr, val_hr = train_test_split(lr_images, hr_images, test_size=0.2, random_state=42)

# Define SRCNN Model
model = Sequential([
    Conv2D(64, (9, 9), activation='relu', padding='same', input_shape=(128, 128, 3)),
    Conv2D(32, (1, 1), activation='relu', padding='same'),
    Conv2D(3, (5, 5), activation='linear', padding='same')
])

# Compile Model
model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

# Train Model
history = model.fit(train_lr, train_hr, epochs=20, batch_size=16, validation_data=(val_lr, val_hr))

# Save Model
model.save("super_resolution_model.h5")

print("✅ Model Training Completed and Saved!")
