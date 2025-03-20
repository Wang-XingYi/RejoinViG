## RejoinViG

# Overview
This repository contains the source code for Reassembling the Scattered Civilization: An Efficient Contour, Script, and Texture-Aware Rejoining of Fragment Images from Ancient Bone Stick

# Usage

## Installation 

```
conda install pytorch==2.1.1 torchvision==0.16.1
```
```
pip install -r requirements.txt
```
### Synthetic Dataset
#### 1. Generate fragment images
./data/01_fragImge.py: Through horizontal and vertical split curves, the bone stick images are divided into fragment images corresponding to top-bottom, bottom-top, left-right, and right-left rejoining.

./data/02_dele_background.py: Crop the segmented bone sticks, resizing the 1200×1170 image to 800×800 to remove a significant amount of background information.
./data/03_remove_small_part_img.py: Some fragments obtained after horizontal splitting have very short matching edges. Images with such short edges should not be used for top-bottom rejoining, so we need to remove them.
./data/04_delet_small_dataset.py: Remove records from the generated rejoinable txt document that contain images with too short matching edges (for both top-bottom and bottom-top rejoining). Additionally, some segmented fragments have discontinuous 
matching edges, and such images should not appear in the rejoinable txt file.
#### 2. Dataset division
./data/05_divide_dataset.py: Divide the dataset into training, test, and validation sets
./data/10_resize_train_val_test_imgs.py: Resize all images in the dataset (Train, Test, and validation) to 224×224 size.
#### 3. Generate the txt file for the dataset
./data/06_create_train_val_not_rejoin_txt.py: Generate unrejoinable data for the training and validation set. 
./data/07_train_val_txt.py: Generate txt files for train and validation set
./data/08_create_test_not_rejoin_full_txt.py: Generate unrejoinable data for the test set
./data/09_creat_test_full_txt.py: Generate txt files for test set

### Train RejoinViG:
```
python main.py
```

### Test RejoinViG:
```
python test.py
python results_evaluate.py
```



