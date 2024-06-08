import os
import cv2
import numpy as np
from collections import Counter

# Function to extract unique RGB codes from an image
def extract_unique_colors(image_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    # Reshape the image to a 2D array of pixels
    pixels = image_rgb.reshape(-1, 3)
    # Use a Counter to find unique colors
    unique_colors = Counter(map(tuple, pixels))
    return unique_colors

# Function to walk through a directory and process all images
def get_all_unique_colors_from_folder(folder_path):
    all_unique_colors = set()
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                image_path = os.path.join(root, file)
                unique_colors = extract_unique_colors(image_path)
                all_unique_colors.update(unique_colors.keys())
    return all_unique_colors

# Specify the folder containing your masked images
folder_path = r'D:\JIO Institute\Capstone\Capstone_V2\ID_code_UnetR\Multiclass-Image-Segmentation-using-UNETR-in-TensorFlow\casa_data_new\label_imgs_v2\train'

# Get all unique RGB colors from the folder
unique_colors_set = get_all_unique_colors_from_folder(folder_path)

# Convert the set to a sorted list
rgb_codes = sorted(list(unique_colors_set))

print("RGB Codes:", rgb_codes)
