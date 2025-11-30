"""
Remove records from the generated rejoinable txt document that contain images with too short matching edges 
(for both top-bottom and bottom-top rejoining). Additionally, some split fragments have discontinuous 
matching edges, and such images should not appear in the rejoinable txt file.

"""

import os

def delet_delete_top_bottom_rejoin(delete_list):
    # Create a new list to store the results
    new_list = []

    for item in delete_list:
        # Obtain the corresponding top-bottom rejoining items based on the image name
        if '_11' in item:
            new_item = item + ' ' + item.replace('_11', '_2')
        elif '_12' in item:
            new_item = item + ' ' + item.replace('_12', '_2')
        elif '_21' in item:
            new_item = item + ' ' + item.replace('_21', '_1')
        elif '_22' in item:
            new_item = item + ' ' + item.replace('_22', '_1')
        new_list.append(new_item)
        # Split into two parts using a space
        first_part, second_part = new_item.split()
        # Swap the positions
        swapped_item = f"{second_part} {first_part}"
        # Append the swapped item; corresponding bottom-top rejoining should also be removed
        new_list.append(swapped_item)
    return new_list

if __name__ == '__main__':
    # These images have very short matching edges and should not appear in the dataset.
    fragments_delete_part_path=r'./02-fragments_delete_part'
    delete_fragments_top_bottom_rejoin=os.listdir(fragments_delete_part_path)
    # This folder contains images where the matching edges are discontinuous after splitting.
    # Therefore, such images should not appear in the dataset.
    fragments_delete='./02-fragments_delete'
    delete_fragments_rejoin=os.listdir(fragments_delete)

    # Obtain the top-bottom and bottom-top rejoining items that need to be removed from the txt file
    new_list=delet_delete_top_bottom_rejoin(delete_fragments_top_bottom_rejoin)

    print(new_list)


    # File path of the original rejoining records
    input_file_path = 'labels_rejoining_log.txt'
    # File path of the rejoining records after deletion
    output_file_path = 'labels_rejoining_log_dele.txt'
    with open(input_file_path, 'r') as file:
        lines = file.readlines()

    # Remove the top-bottom and bottom-top rejoining items that need to be deleted from the original rejoining record
    filtered_lines = [line for line in lines if not any(item in line for item in new_list)]

    # Remove the items with discontinuous matching edges from the original rejoining record
    filtered_lines_2 = [line for line in filtered_lines if not any(item in line for item in delete_fragments_rejoin)]


    with open(output_file_path, 'w') as file:
        file.writelines(filtered_lines_2)

    print(f"删除后的文件已保存为 {output_file_path}")


