"""
Some fragments obtained after horizontal splitting have very short matching edges.
Images with such short edges should not be used for top-bottom rejoinable,
so we need to remove them. The images marked for deletion are saved into a new folder.
Later, we manually review these images to verify if any actually have relatively long edges.
If they do, such images can be retained (i.e., removed from the deletion folder).

"""

import os

import cv2
import numpy as np

if __name__ == '__main__':
    # Source folder
    dir_path=r'../data/02-fragments-2_crop'

    # Folder for saving images that need to be deleted
    remove_path=r'./02-fragments_delete_part'
    if not os.path.exists(remove_path):
        os.makedirs(remove_path)
    for img_name in os.listdir(dir_path):
        if '_1.bmp' in img_name or '_2.bmp' in img_name:
            continue
        img = cv2.imread(os.path.join(dir_path,img_name))
        h, w = img.shape[:2]
        non_bg_mask = np.any(img < (80, 80, 80), axis=-1)
        non_bg_coords = np.argwhere(non_bg_mask)
        if non_bg_coords.size == 0:
            continue

        y_min, x_min = non_bg_coords.min(axis=0)
        y_max, x_max = non_bg_coords.max(axis=0)

        distance=x_max-x_min
        if distance < 160:
            cv2.imwrite(os.path.join(remove_path,img_name),img)

            print(f'Removed {img_name} : {distance}')
