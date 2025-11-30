"""
Through horizontal and vertical splitting curves, the bone stick images are divided into fragment images 
corresponding to top-bottom, bottom-top, left-right, and right-left rejoinable.
"""

import random
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os



def generate_horizontal_curve(img_shape, previous_y=None, num_points=30, max_amplitude=10, slope_amplitude=0.1):
    """Generate a horizontal split curve to divide the image into top and bottom parts."""
    height, width = img_shape[:2]
    # num_points = np.random.randint(20, 50)
    num_points = np.random.randint(10, 30)
    x_points = np.linspace(0, width, num=num_points)
    slope_amplitude = np.random.uniform(-0.3, 0.3)

    if previous_y is not None:
        y_offset = previous_y
    else:
        y_offset = height // 2

    center_slope = slope_amplitude * (height / width)
    y_center_line = center_slope * x_points + y_offset
    y_points = y_center_line + np.random.randint(-max_amplitude, max_amplitude, size=num_points)

    y_points[0] = np.clip(y_points[0], 0, height)
    y_points[-1] = np.clip(y_points[-1], 0, height)

    return x_points, y_points, y_center_line[0]
def generate_vertical_curve(img_shape, previous_x=None, num_points=30, max_amplitude=10, slope_amplitude=0.1):
    """Generate a vertical split curve to divide the image into left and right parts."""
    height, width = img_shape[:2]
    # num_points = np.random.randint(20, 50)
    num_points = np.random.randint(10, 30)
    y_points = np.linspace(0, height, num=num_points)
    slope_amplitude = np.random.uniform(-0.1, 0.1)
    # slope_amplitude = 0.1
    print(slope_amplitude)

    if previous_x is not None:
        x_offset = previous_x
    else:
        x_offset = width // 2

    center_slope = slope_amplitude * (height / width)
    x_center_line = center_slope * y_points + x_offset
    x_points = x_center_line + np.random.randint(-max_amplitude, max_amplitude, size=num_points)

    x_points[0] = np.clip(x_points[0], 0, height)
    x_points[-1] = np.clip(x_points[-1], 0, height)

    return x_points, y_points, None

def apply_random_disturbance(curve, disturbance_range=15):
    """ Apply larger random perturbations to the y-coordinates of the curve """
    disturbed = curve + np.random.randint(-disturbance_range, disturbance_range, size=curve.shape)
    return disturbed
def calcu_split_img(image,curve_x,curve_y, bg_color=(255, 255, 255),):
    height, width = image.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)

    curve_points = np.array([curve_x, curve_y]).T.astype(np.int32)
    left_polygon = np.vstack(([0, 0], curve_points, [width, height], [0, height])).astype(np.int32)
    cv2.fillPoly(mask, [left_polygon], 255)

    bg_color = np.array(bg_color, dtype=np.uint8)
    colored_bg = np.ones_like(image) * bg_color

    img1 = np.where(mask[:, :, np.newaxis] == 255, image, colored_bg)
    mask_inv = cv2.bitwise_not(mask)
    img2 = np.where(mask_inv[:, :, np.newaxis] == 255, image, colored_bg)

    return img1, img2

def split_image(image, curve_x, curve_y, bg_color=(255, 255, 255)):
    img1,img2=calcu_split_img(image, curve_x, curve_y, bg_color)
    # 随机选择分割方向
    direction = 'img1' if np.random.rand() < 0.5 else 'img2'
    # 对 img1 的曲线进行更大幅度的扰动，模拟缀合误差
    disturbed_curve_y = apply_random_disturbance(curve_y, disturbance_range=5)
    img11, img22 = calcu_split_img(image, curve_x, disturbed_curve_y, bg_color)
    if direction=='img1':
        return img11,img2
    else:
        return img1, img22



def is_not_empty(img, bg_color, min_area_ratio=0.02):
    non_bg_pixels = np.sum(np.any(img != bg_color, axis=-1))
    total_pixels = img.shape[0] * img.shape[1]
    return non_bg_pixels / total_pixels > min_area_ratio


def center_image(content_img, bg_color=(255, 255, 255)):
    h, w = content_img.shape[:2]
    non_bg_mask = np.any(content_img != bg_color, axis=-1)
    non_bg_coords = np.argwhere(non_bg_mask)

    if non_bg_coords.size == 0:
        return np.ones_like(content_img) * bg_color

    y_min, x_min = non_bg_coords.min(axis=0)
    y_max, x_max = non_bg_coords.max(axis=0)

    cropped = content_img[y_min:y_max + 1, x_min:x_max + 1]

    result = np.ones_like(content_img) * bg_color
    y_offset = (h - cropped.shape[0]) // 2
    x_offset = (w - cropped.shape[1]) // 2

    result[y_offset:y_offset + cropped.shape[0], x_offset:x_offset + cropped.shape[1]] = cropped
    return result

def cal_imgs(img,curve_x,curve_y,bg_color,min_area_ratio):
    img1, img2 = split_image(img, curve_x, curve_y, bg_color)
    img1_centered =  np.zeros_like(img1)
    img2_centered =  np.zeros_like(img1)
    if is_not_empty(img1, bg_color, min_area_ratio):
        img1_centered = center_image(img1, bg_color)
    if is_not_empty(img2, bg_color, min_area_ratio):
        img2_centered = center_image(img2, bg_color)
    return  img1_centered,img2_centered

def iterative_split(image,save_path, img_name,log,bg_color=(255, 255, 255), min_area_ratio=0.02):
    final_images = []  # Retain only the final fragments.
    curve_x, curve_y, y_offset = generate_horizontal_curve(image.shape, previous_y=None)
    img1,img2=cal_imgs(image,curve_x,curve_y,bg_color,min_area_ratio)
    is_vertical_curve=np.random.randint(0,4)
    print(is_vertical_curve)
    img11 = img12 = img21 = img22 = np.zeros_like(img1)
    if np.any(img1):
        final_images.append(img1)
        save_images(img1,save_path+img_name+'_1.bmp')
        curve1_x, curve1_y, y_offset = generate_vertical_curve(img1.shape)
        img11, img12 = cal_imgs(img1, curve1_x, curve1_y,bg_color,min_area_ratio)

    if np.any(img2):
        final_images.append(img2)
        save_images(img2, save_path+img_name + '_2.bmp')
        curve2_x, curve2_y, y_offset = generate_vertical_curve(img2.shape)
        img21, img22 = cal_imgs(img2, curve2_x, curve2_y,bg_color,min_area_ratio)
    # save format: img_source img_target top_bottom bottom_top left_right right_left not_rejoining
    if np.any(img1) and np.any(img2):
        log.write(f'{img_name}_1.bmp {img_name}_2.bmp 1 0 0 0 0\n')
        log.write(f'{img_name}_2.bmp {img_name}_1.bmp 0 1 0 0 0\n')

    if is_vertical_curve == 0: # Do not perform left-right symmetry
        return final_images
    elif is_vertical_curve==1: # Perform left-right symmetry on img1
        if np.any(img11):
            final_images.append(img11)
            save_images(img11, save_path + img_name+'_11.bmp')
            if np.any(img2):
                log.write(f'{img_name}_2.bmp {img_name}_11.bmp 0 1 0 0 0\n')
                log.write(f'{img_name}_11.bmp {img_name}_2.bmp 1 0 0 0 0\n')
        if np.any(img12):
            final_images.append(img12)
            save_images(img12, save_path + img_name+'_12.bmp')
            if np.any(img2):
                log.write(f'{img_name}_2.bmp {img_name}_12.bmp 0 1 0 0 0\n')
                log.write(f'{img_name}_12.bmp {img_name}_2.bmp 1 0 0 0 0\n')
        if np.any(img11) and np.any(img12):
            log.write(f'{img_name}_11.bmp {img_name}_12.bmp 0 0 0 1 0\n')
            log.write(f'{img_name}_12.bmp {img_name}_11.bmp 0 0 1 0 0\n')
        return final_images
    elif is_vertical_curve==2: # Perform left-right symmetry on img2
        if np.any(img21):
            final_images.append(img21)
            save_images(img21, save_path + img_name+'_21.bmp')
            if np.any(img1):
                log.write(f'{img_name}_1.bmp {img_name}_21.bmp 1 0 0 0 0\n')
                log.write(f'{img_name}_21.bmp {img_name}_1.bmp 0 1 0 0 0\n')
        if np.any(img22):
            final_images.append(img22)
            save_images(img22, save_path + img_name+'_22.bmp')
            if np.any(img1):
                log.write(f'{img_name}_1.bmp {img_name}_22.bmp 1 0 0 0 0\n')
                log.write(f'{img_name}_22.bmp {img_name}_1.bmp 0 1 0 0 0\n')
        if np.any(img21) and np.any(img22):
            log.write(f'{img_name}_21.bmp {img_name}_22.bmp 0 0 0 1 0\n')
            log.write(f'{img_name}_22.bmp {img_name}_21.bmp 0 0 1 0 0\n')
        return final_images
    else: # Perform left-right symmetry on img1 and img2
        if np.any(img11):
            final_images.append(img11)
            save_images(img11, save_path + img_name+'_11.bmp')
            if np.any(img2):
                log.write(f'{img_name}_2.bmp {img_name}_11.bmp 0 1 0 0 0\n')
                log.write(f'{img_name}_11.bmp {img_name}_2.bmp 1 0 0 0 0\n')
        if np.any(img12):
            final_images.append(img12)
            save_images(img12, save_path + img_name+'_12.bmp')
            if np.any(img2):
                log.write(f'{img_name}_2.bmp {img_name}_12.bmp 0 1 0 0 0\n')
                log.write(f'{img_name}_12.bmp {img_name}_2.bmp 1 0 0 0 0\n')
        if np.any(img11) and np.any(img12):
            log.write(f'{img_name}_11.bmp {img_name}_12.bmp 0 0 0 1 0\n')
            log.write(f'{img_name}_12.bmp {img_name}_11.bmp 0 0 1 0 0\n')
        if np.any(img21):
            final_images.append(img21)
            save_images(img21, save_path + img_name+'_21.bmp')
            if np.any(img1):
                log.write(f'{img_name}_1.bmp {img_name}_21.bmp 1 0 0 0 0\n')
                log.write(f'{img_name}_21.bmp {img_name}_1.bmp 0 1 0 0 0\n')
        if np.any(img22):
            final_images.append(img22)
            save_images(img22, save_path + img_name+'_22.bmp')
            if np.any(img1):
                log.write(f'{img_name}_1.bmp {img_name}_22.bmp 1 0 0 0 0\n')
                log.write(f'{img_name}_22.bmp {img_name}_1.bmp 0 1 0 0 0\n')
        if np.any(img21) and np.any(img22):
            log.write(f'{img_name}_21.bmp {img_name}_22.bmp 0 0 0 1 0\n')
            log.write(f'{img_name}_22.bmp {img_name}_21.bmp 0 0 1 0 0\n')
        return final_images



def save_images(img, save_path):
    img = img.astype(np.uint8)
    cv2.imwrite(save_path, img)



def main(image_path, save_path,img_name,log, bg_color=(255, 255, 255), min_area_ratio=0.02):
    image = cv2.imread(image_path)
    images = iterative_split(image, save_path,img_name,log,bg_color, min_area_ratio=min_area_ratio)

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
if __name__ == '__main__':
    set_seed(3)
    # source file
    dir_path=r'../data/01-dataset_modify_background'
    min_area_ratio = 0.01  # Minimum area ratio of effective content
    bg_color = (255, 255, 255)  # Specify the background colour, e.g. white
    log = open(r'./labels_rejoining_log.txt', 'w', encoding='utf-8')
    log.write('img_source img_target top_bottom bottom_top left_right right_left not_rejoining\n')
    for img in os.listdir(dir_path):
        print(img)
        save_path='./01-fragments/'
        image_path=os.path.join(dir_path,img)
        img_name=img.split('.')[0]
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        main(image_path, save_path, img_name, log, bg_color, min_area_ratio)





