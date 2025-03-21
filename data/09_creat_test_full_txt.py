import os
"""
Generate txt files for test set
"""

def read_log_file(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    return lines

def save_to_file(file_path, content):
    with open(file_path, 'w') as f:
        for line in content:
            f.write(line)

def process_files(folder_name, log_file, rejoin_log, output_file):
    folder_files = set(os.listdir(folder_name))

    rejoin_lines = read_log_file(rejoin_log)

    selected_lines = [line for line in rejoin_lines if line.split(' ')[0] in folder_files]

    log_lines = read_log_file(log_file)
    flag=0
    for i in range(len(selected_lines)):
        print(i)
        item = selected_lines[i].split(' ')[0] + ' ' + selected_lines[i].split(' ')[1]
        k=True
        for j in range(len(log_lines)):
            if item.lower() in log_lines[j]:
                log_lines[j]=selected_lines[i]
                k=False
        if k:
            print(item)



    save_to_file(output_file, log_lines)

def main():
    # source file
    test_folder = './Test'
    # unrejoinable file
    test_log = './Test_labels_not_rejoining_log.txt'
    # rejoinable file
    rejoin_log = './labels_rejoining_log_dele.txt'
    # output file
    test_output = './Test_full.txt'

    process_files(test_folder, test_log, rejoin_log, test_output)


if __name__ == '__main__':
    main()
