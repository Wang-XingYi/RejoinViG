"""
Automatically generate rejoinable fragment combinations that can form a complete bone stick based on Test.
such as:
00002_1.bmp 00002_2.bmp
00002_11.bmp 00002_12.bmp 00002_2.bmp
00020_1.bmp 00020_2.bmp
"""

import os
from collections import defaultdict



root_dir = r"../Dataset/Test"

# group files by the prefix before "_"
groups = defaultdict(list)
for fname in os.listdir(root_dir):
    full_path = os.path.join(root_dir, fname)
    if not os.path.isfile(full_path):
        continue
    if "_" not in fname:
        continue
    prefix = fname.split("_", 1)[0]
    groups[prefix].append(fname)


seen_pairs = set()
for prefix, files in groups.items():
    suffix_to_files = defaultdict(list)
    for f in files:
        name_no_ext, ext = os.path.splitext(f)    # 00002_1.bmp -> (00002_1, .bmp)
        _, suffix = name_no_ext.split("_", 1)     # "00002_1" -> ["00002", "1"]
        suffix_to_files[suffix].append(f)

    # record rejoinable fragment combinations
    pair=tuple()
    if "1" in suffix_to_files and "2" in suffix_to_files:
        pair = tuple(sorted((suffix_to_files["1"][0], suffix_to_files["2"][0])))
    seen_pairs.add(pair)

    if "1" in suffix_to_files and "21" in suffix_to_files and "22" in suffix_to_files:
        pair = tuple(sorted((suffix_to_files["1"][0], suffix_to_files["21"][0],suffix_to_files["22"][0])))
    elif "1" in suffix_to_files and "21" in suffix_to_files:
        pair = tuple(sorted((suffix_to_files["1"][0], suffix_to_files["21"][0])))
    elif "1" in suffix_to_files and "22" in suffix_to_files:
        pair = tuple(sorted((suffix_to_files["1"][0], suffix_to_files["22"][0])))
    seen_pairs.add(pair)

    if "2" in suffix_to_files and "11" in suffix_to_files and "12" in suffix_to_files:
        pair = tuple(sorted((suffix_to_files["2"][0], suffix_to_files["11"][0],suffix_to_files["12"][0])))
    elif "2" in suffix_to_files and "11" in suffix_to_files:
        pair = tuple(sorted((suffix_to_files["2"][0], suffix_to_files["11"][0])))
    elif "2" in suffix_to_files and "12" in suffix_to_files:
        pair = tuple(sorted((suffix_to_files["2"][0], suffix_to_files["12"][0])))
    seen_pairs.add(pair)

    if "11" in suffix_to_files and "12" in suffix_to_files  and "21" in suffix_to_files and "22" in suffix_to_files:
        pair = tuple(sorted((suffix_to_files["11"][0],suffix_to_files["12"][0],
                             suffix_to_files["21"][0],suffix_to_files["22"][0])))



    seen_pairs.add(pair)

with open('../Dataset/GT.txt', 'w', encoding='utf-8') as f:

    for tup in sorted(seen_pairs):
        line = " ".join(tup)
        f.write(line + "\n")

print("Saved successfully")

