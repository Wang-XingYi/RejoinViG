import os
from PIL import Image

"""
Resize all images in the dataset (train, test, and validation) to 224×224 size.
"""


folders = {
    "Train": "./Train",
    "Test": "./Test",
    "Val": "./Val",
}


target_size = (224, 224)


def process_and_save_images(folder_path):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif')):
                file_path = os.path.join(root, file)


                with Image.open(file_path) as img:

                    img_resized = img.resize(target_size, Image.Resampling.LANCZOS)

                    img_resized.save(file_path)



for folder_name, folder_path in folders.items():
    print(f"Processing folder: {folder_name}")
    process_and_save_images(folder_path)
    print(f"Processed and saved: {folder_name}")
