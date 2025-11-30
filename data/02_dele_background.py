"""
Crop the split bone sticks, resizing the 1200×1170 image to 800×800 to 
remove a significant amount of background information.
"""

import os

import cv2
import numpy as np


def center_crop_image(content_img, output_size=(800, 800), bg_color=(255, 255, 255), tolerance=5):
    h, w = content_img.shape[:2]

    # Identify the non-background region using a tolerance value
    mask_r = np.abs(content_img[:, :, 0] - bg_color[0]) > tolerance
    mask_g = np.abs(content_img[:, :, 1] - bg_color[1]) > tolerance
    mask_b = np.abs(content_img[:, :, 2] - bg_color[2]) > tolerance

    # Merge the mask
    non_bg_mask = mask_r | mask_g | mask_b

    # Find coordinates of non-background pixels
    non_bg_coords = np.argwhere(non_bg_mask)

    if non_bg_coords.size > 0:
        # Find the bounding box of the non-background region
        y_min, y_max = non_bg_coords[:, 0].min(), non_bg_coords[:, 0].max()
        x_min, x_max = non_bg_coords[:, 1].min(), non_bg_coords[:, 1].max()

        # Crop the non-background part of the image
        cropped = content_img[y_min:y_max + 1, x_min:x_max + 1]

        # Get the dimensions of the cropped image
        cropped_h, cropped_w = cropped.shape[:2]

        # Create a white background image of size 800x800
        result = np.ones((output_size[0], output_size[1], 3), dtype=np.uint8) * 255

        # f the cropped image size is larger than 800x800, crop the extra parts
        start_y = max(0, (cropped_h - output_size[0]) // 2)
        start_x = max(0, (cropped_w - output_size[1]) // 2)

        end_y = min(cropped_h, start_y + output_size[0])
        end_x = min(cropped_w, start_x + output_size[1])

        cropped = cropped[start_y:end_y, start_x:end_x]

        #  Center the cropped image on the 800x800 background
        y_offset = (output_size[0] - cropped.shape[0]) // 2
        x_offset = (output_size[1] - cropped.shape[1]) // 2

        result[y_offset:y_offset + cropped.shape[0], x_offset:x_offset + cropped.shape[1]] = cropped

        return result
    else:
        # If no non-background region is found, return a white image
        return np.ones((output_size[0], output_size[1], 3), dtype=np.uint8) * 255




if __name__ == '__main__':
    dir_path=r'./01-fragments'
    save_path=r'./02-fragments_crop'
    for img_name in os.listdir(dir_path):
        content_img = cv2.imread(os.path.join(dir_path, img_name))
        # Call the function to crop and center the image
        result_img = center_crop_image(content_img)



        # Save the processed image
        cv2.imwrite(os.path.join(save_path,img_name), result_img)
        print(img_name)


