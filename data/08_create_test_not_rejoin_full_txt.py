"""
Generate unrejoinable data for the test set
"""
import os

if __name__ == '__main__':
    path=r'./Test'
    imgs=os.listdir(path)
    f=open('Test_labels_not_rejoining_log.txt', 'w')
    for i in range(len(imgs)):
        for j in range(len(imgs)):
            if imgs[i]!=imgs[j]:
                f.write(f"{imgs[i]} {imgs[j]} 0 0 0 0 1\n")


