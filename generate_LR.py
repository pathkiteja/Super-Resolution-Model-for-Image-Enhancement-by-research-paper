import os
import cv2
from tqdm import tqdm

# Define paths (Update the paths according to your folder structure)
hr_dir = "DIV2K/DIV2K_train_HR/"  # High-Resolution Images
lr_dir = "DIV2K/DIV2K_train_LR/"  # Folder to save Low-Resolution Images

# Create LR directory if it doesn't exist
if not os.path.exists(lr_dir):
    os.makedirs(lr_dir)

# Downscale HR images to generate LR images
scale = 4  # Downscaling factor
for img_name in tqdm(os.listdir(hr_dir)):
    img_path = os.path.join(hr_dir, img_name)
    lr_img_path = os.path.join(lr_dir, img_name)

    # Read the image
    img = cv2.imread(img_path)
    if img is not None:
        h, w = img.shape[:2]
        lr_img = cv2.resize(img, (w//scale, h//scale), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(lr_img_path, lr_img)

print("✅ Low-resolution images generated successfully!")
