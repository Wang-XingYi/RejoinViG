import os
import shutil
import random

import numpy as np
import torch

"""
Divide the dataset into training, test, and validation sets.
"""
def split_images_and_copy(source_folder, train_folder, test_folder, val_folder, image_extensions=('.jpg', '.png', '.bmp')):

    for folder in [train_folder, test_folder, val_folder]:
        if not os.path.exists(folder):
            os.makedirs(folder)

    # Group images using the part before '_' as the key
    groups = {}

    # Traverse the source folder to get all image files
    for image_file in os.listdir(source_folder):
        if image_file.lower().endswith(image_extensions):
            group_key = image_file.split('_')[0]
            if group_key not in groups:
                groups[group_key] = []
            groups[group_key].append(image_file)

    grouped_images = list(groups.values())

    # Shuffle the groups to ensure random assignment
    random.shuffle(grouped_images)

    num_groups = len(grouped_images)
    train_split = int(num_groups * 0.9)
    test_split = int(num_groups * 0.95)

    train_groups = grouped_images[:train_split]
    test_groups = grouped_images[train_split:test_split]
    val_groups = grouped_images[test_split:]

    # Ensure no duplicates among the three folders
    def copy_images(image_groups, destination_folder):
        for group in image_groups:
            for image_file in group:
                source_file_path = os.path.join(source_folder, image_file)
                destination_file_path = os.path.join(destination_folder, image_file)
                if not os.path.exists(destination_file_path):  # 确保文件未存在
                    shutil.copy(source_file_path, destination_file_path)

    # Copy images to the Train folder
    copy_images(train_groups, train_folder)
    # Copy images to the Test folder
    copy_images(test_groups, test_folder)
    # Copy images to the Val folder
    copy_images(val_groups, val_folder)

    print(f"Train:{len(train_groups)}, Test:{len(test_groups)}, Val:{len(val_groups)}")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
def main():
    set_seed(3)
    # Folder where the source images are located
    source_folder = './02-fragments_crop'
    # Target folders for Train, Test, and Val
    train_folder = './Train'
    test_folder = './Test'
    val_folder = './Val'

    # Call the function to divide  and copy images into groups
    split_images_and_copy(source_folder, train_folder, test_folder, val_folder)


if __name__ == '__main__':
    main()
