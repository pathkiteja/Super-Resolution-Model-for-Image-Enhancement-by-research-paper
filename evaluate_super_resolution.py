import numpy as np
import tensorflow as tf
import cv2
import os
import matplotlib.pyplot as plt
import random

# Load trained model
custom_objects = {"mse": tf.keras.losses.MeanSquaredError()}
model = tf.keras.models.load_model("super_resolution_model.h5", custom_objects=custom_objects)

# Define paths
lr_dir = "DIV2K/DIV2K_train_LR/"
hr_dir = "DIV2K/DIV2K_train_HR/"

# Pick a random image
img_name = random.choice(os.listdir(lr_dir))  # Selects a different image every time
lr_img_path = os.path.join(lr_dir, img_name)
hr_img_path = os.path.join(hr_dir, img_name)

# Load images
lr_img = cv2.imread(lr_img_path)
hr_img = cv2.imread(hr_img_path)

# Preprocess for model input
lr_img = cv2.resize(lr_img, (128, 128)) / 255.0
lr_img_input = np.expand_dims(lr_img, axis=0)

# Generate Super-Resolution Image
sr_img = model.predict(lr_img_input)
sr_img = np.clip(sr_img[0], 0, 1)

# Convert images to uint8 for visualization
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(cv2.cvtColor((lr_img * 255).astype(np.uint8), cv2.COLOR_BGR2RGB))
axes[0].set_title("Low-Resolution")
axes[1].imshow(cv2.cvtColor((hr_img * 255).astype(np.uint8), cv2.COLOR_BGR2RGB))
axes[1].set_title("Original High-Resolution")
axes[2].imshow((sr_img * 255).astype(np.uint8))
axes[2].set_title("Super-Resolution Output")

plt.show()
