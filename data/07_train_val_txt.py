"""
Generate txt files for train and validation set
"""

import os
import random

def read_log_file(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    return lines[1:]

def save_to_file(file_path, content):
    with open(file_path, 'w') as f:
        for line in content:
            f.write(line)
# Merge rejoinable and non-rejoinable data into one file
def process_files(folder_name, log_file, rejoin_log, output_file):

    folder_files = set(os.listdir(folder_name))

    rejoin_lines = read_log_file(rejoin_log)


    selected_lines = [line for line in rejoin_lines if line.split(' ')[0] in folder_files]

    log_lines = read_log_file(log_file)
    random.shuffle(log_lines)

    # Merge the two parts of the content
    final_content = selected_lines + log_lines


    random.shuffle(final_content)

    save_to_file(output_file, final_content)

def main():
    # source file
    train_folder = './Train'
    val_folder = './Val'

    # unrejoinable file
    train_log = './Train_labels_not_rejoining_log.txt'
    val_log = './Val_labels_not_rejoining_log.txt'

    # rejoinable file
    rejoin_log = './labels_rejoining_log_dele.txt'

    # output file
    train_output = './Train.txt'
    val_output = './Val.txt'


    process_files(train_folder, train_log, rejoin_log, train_output)
    process_files(val_folder, val_log, rejoin_log, val_output)

if __name__ == '__main__':
    main()

