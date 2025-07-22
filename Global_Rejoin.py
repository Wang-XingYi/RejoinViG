from collections import defaultdict

file = r'11_RejoinViG-top10.txt'

# -------------------- 1. Read data --------------------
edges = []
with open('../log/'+file, "r") as f:
    for line in f:
        src, tgt, pre_class, label = line.strip().split()
        if pre_class == 'pre_classes':
            continue
        edges.append((src, tgt, int(pre_class)))
edges_set = set(edges)

# -------------------- 2. Build graph structure --------------------
filter_map = defaultdict(list)
for src, tgt, pre_class in edges:
    if pre_class == 1 or pre_class == 3:
        target_class = pre_class - 1
    else:
        target_class = pre_class + 1
    if (tgt, src, target_class) in edges:
        filter_map[src].append((tgt, pre_class))

# -------------------- 3. DFS: Only save paths when a cycle is formed --------------------
def dfs_all_paths_unique_class(filter_map, current, start, visited, used_classes, current_path, all_paths):
    visited.append(current)

    for target, pre_class in filter_map.get(current, []):
        if pre_class in used_classes:
            continue
        pre_num = len(current_path) - 2
        if pre_num>=0:
            if (current_path[pre_num][2] == 0 and current_path[pre_num + 1][2] == 1) or (
                    current_path[pre_num][2] == 1 and current_path[pre_num + 1][2] == 0):
                if ((current_path[pre_num][0], current_path[pre_num + 1][1], 2) not in edges_set) or (
                        (current_path[pre_num][0], current_path[pre_num + 1][1], 3) not in edges_set):
                    continue

            if (current_path[pre_num][2] == 2 and current_path[pre_num + 1][2] == 3) or (
                    current_path[pre_num][2] == 3 and current_path[pre_num + 1][2] == 2):
                if ((current_path[pre_num][0], current_path[pre_num + 1][1], 0) not in edges_set) or (
                        (current_path[pre_num][0], current_path[pre_num + 1][1], 1) not in edges_set):
                    continue
        if target in visited:
            if target == start and len(current_path) > 0:
                # A valid loop is found; only save the current path with the closing edge
                all_paths.append(current_path + [(current, target, pre_class)])
            continue

        new_path = current_path + [(current, target, pre_class)]
        new_visited = visited.copy()
        new_used_classes = used_classes.copy()
        new_used_classes.add(pre_class)

        dfs_all_paths_unique_class(
            filter_map,
            target,
            start,
            new_visited,
            new_used_classes,
            new_path,
            all_paths
        )

# -------------------- 4. Main traversal logic --------------------
all_paths = []

for source in filter_map.keys():
    dfs_all_paths_unique_class(
        filter_map,
        current=source,
        start=source,
        visited=[],
        used_classes=set(),
        current_path=[],
        all_paths=all_paths
    )

num = 0
saved_path_sets = set()  # Used to record node sets of already saved paths

with open('./rejoin_results/'+file, 'w', encoding='utf-8') as f:
    for i, path in enumerate(all_paths):
        if len(path) >= 2:
            # Construct a node set for the path (considering only nodes, not edge directions)
            node_set = frozenset([src for src, _, _ in path] + [path[-1][1]])

            if node_set in saved_path_sets:
                continue 

            saved_path_sets.add(node_set) 
            num += 1

            print(f"Path {num}:")
            f.write(f"Path {num}:\n")
            for src, tgt, cls in path:
                print(f"    {src} → {tgt}, pre_class = {cls}")
                f.write(f"    {src} → {tgt}, pre_class = {cls}\n")
            print("-" * 40)
            f.write("-" * 40 + '\n')
