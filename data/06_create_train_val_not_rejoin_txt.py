"""
Generate unrejoinable data for the training and validation set.
Randomly select one image from each group (grouped by the part before "_" in the image name),
and then randomly select one image from a other group to form a pair.
"""

import os
import random

import numpy as np
import torch


def random_select_files(file_dict):
    result_list = []

    for same_class_key, same_class_files in file_dict.items():
        # Randomly choose one file name from the current category
        same_class_file = random.choice(same_class_files)

        # Randomly choose one file name from a ohther category
        different_class_keys = [k for k in file_dict.keys() if k != same_class_key]
        different_class_key = random.choice(different_class_keys)
        different_class_file = random.choice(file_dict[different_class_key])

        # Combine the results and add them to the result list
        result = f"{same_class_file} {different_class_file} 0 0 0 0 1"
        result_list.append(result)

    return result_list

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

def not_rejoin_txt(folder_path,output_file):
    # Read the file names in the folder and categorize them by the part before "_"
    file_dict = {}
    for file_name in sorted(os.listdir(folder_path)):
        if file_name.endswith(".bmp"):
            key = file_name.split('_')[0]  # Use the part before "_" as the category key
            if key not in file_dict:
                file_dict[key] = []
            file_dict[key].append(file_name)

    # Generate the result list
    output_lines = random_select_files(file_dict)

    # Save the results to a txt file
    with open(output_file, 'w') as f:
        f.write('img_source img_target top_bottom bottom_top left_right right_left not_rejoining\n')
        for line in output_lines:
            f.write(line + '\n')
    print(f"save to {output_file}")

if __name__ == '__main__':
    set_seed(3)

    train_path = './Train'
    val_path = './Val'

    # Save the results to txt files
    train_output_file = 'Train_labels_not_rejoining_log.txt'
    val_output_file = 'Val_labels_not_rejoining_log.txt'
    not_rejoin_txt(train_path,train_output_file)
    not_rejoin_txt(val_path,val_output_file)




