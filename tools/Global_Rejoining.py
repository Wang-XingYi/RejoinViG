"""
Global rejoining: generate complete bone sticks based on the Top-K candidate image set.
"""

import os
from collections import defaultdict


def remove_duplicates(all_paths):
    unique_pairs = []
    cleaned_path = []
    for path in all_paths:
        pairs = []
        for item in path:
            pairs.append(item[0])
            pairs.append(item[1])
        sorted_pair = sorted(pairs)

        if sorted_pair not in unique_pairs:
            unique_pairs.append(sorted_pair)
            cleaned_path.append(path)
    return cleaned_path



# DFS
def dfs_all_paths_unique_class(edges_set,filter_map, current, visited, used_classes, current_path, all_paths, max_depth=10):
    if current in visited or max_depth <= 0:
        return

    visited.add(current)

    for target, pre_class in filter_map.get(current, []):
        if target in visited:
            continue
        if pre_class in used_classes:
            continue


        new_path = current_path + [(current, target, pre_class)]
        new_visited = visited.copy()
        new_used_classes = used_classes.copy()
        new_used_classes.add(pre_class)
        # all_paths.append(new_path)

        # save current path
        if len(new_path) > 1:
            pre_num = len(new_path) - 2
            if (new_path[pre_num][2] == 0 and new_path[pre_num + 1][2] == 1) or (
                    new_path[pre_num][2] == 1 and new_path[pre_num + 1][2] == 0):
                if ((new_path[pre_num][0], new_path[pre_num + 1][1], 2) in edges_set) or (
                        (new_path[pre_num][0], new_path[pre_num + 1][1], 3) in edges_set):
                    all_paths.append(new_path)
                else:
                    continue

            if (new_path[pre_num][2] == 2 and new_path[pre_num + 1][2] == 3) or (
                    new_path[pre_num][2] == 3 and new_path[pre_num + 1][2] == 2):
                if ((new_path[pre_num][0], new_path[pre_num + 1][1], 0) in edges_set) or (
                        (new_path[pre_num][0], new_path[pre_num + 1][1], 1) in edges_set):
                    all_paths.append(new_path)
                else:
                    continue
        else:
            all_paths.append(new_path)


        dfs_all_paths_unique_class(
            edges_set,
            filter_map,
            target,
            new_visited,
            new_used_classes,
            new_path,
            all_paths,
            max_depth - 1
        )




def find(file,save_path):
    # read data
    edges = []
    with open('../logs/'+file, "r") as f:
        for line in f:
            src, tgt, pre_class, label = line.strip().split()
            if pre_class == 'pre_classes':
                continue
            edges.append((src, tgt, int(pre_class)))
    edges_set = set(edges)

    # constructing graph structure
    filter_map = defaultdict(list)
    for src, tgt, pre_class in edges:
        if pre_class == 1 or pre_class == 2:
            target_class = pre_class - 1
        else:
            target_class = pre_class + 1
        if (tgt, src, target_class) in edges:
            filter_map[src].append((tgt, pre_class))




    all_paths = []
    for source in filter_map.keys():
        dfs_all_paths_unique_class(
            edges_set,
            filter_map,
            current=source,
            visited=set(),
            used_classes=set(),
            current_path=[],
            all_paths=all_paths,
            max_depth=10
        )

        
    cleaned_path = remove_duplicates(all_paths)

    with open('../Dataset/GT.txt', "r", encoding="utf-8") as f:
        lines = f.readlines()


    GT_lines = [line.strip() for line in lines]

    # print all path
    num=0
    find_num=0
    save_path=os.path.join(save_path, file)
    with open(save_path, 'w', encoding='utf-8') as f:
        for i, path in enumerate(cleaned_path):
            if len(path)>0:
                num+=1
                print(f"Path {num}:")
                f.write(f"Path {num}:\n")
                names = set()
                for src, tgt, cls in path:
                    print(f"    {src} → {tgt}, pre_class = {cls}")
                    f.write(f"    {src} → {tgt}, pre_class = {cls}\n")
                    names.add(src)
                    names.add(tgt)
                print("-" * 40)
                f.write("-" * 40+'\n')

                sorted_names = sorted(names)
                line_str = " ".join(sorted_names)
                if line_str in GT_lines:
                    find_num+=1
    with open(save_path, 'a', encoding='utf-8') as f:
        print(f'number of GT：{len(GT_lines)}, number of finds：{find_num}')
        f.write(f'number of GT：{len(GT_lines)}, number of finds：{find_num}')

if __name__ == '__main__':
    model_name = r'RejoinViG'
    save_path='../rejoin_results'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    K_values = [1, 3, 5, 10, 15, 20]
    for K in K_values:
        file=model_name+'_top'+str(K)+'.txt'
        find(file,save_path)