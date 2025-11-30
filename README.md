# RejoinViG

# Overview
This repository contains the source code forRejoining Precious Artifacts: Efficiently Bone Stick Rejoining Based Massive Fragment Images by Contour, Script, and Texture
![image](https://github.com/Wang-XingYi/RejoinViG/blob/main/Images/network.jpg)

## News

[2025/11/8]: **RejoinViG** has been accepted by **AAAI 2026**. 🔥🔥🔥

## 🚀 Updates
[2025/11/8]: Update code of **RejoinViG**.

# Usage

## Installation 
- Python 3.10.13
```
conda install pytorch==2.1.1 torchvision==0.16.1
```
```
pip install -r requirements.txt
```

### Train RejoinViG:
```
python main.py
```

### Test RejoinViG:
- Test
```
python results_evaluate.py
```
- Calculate Top-K accuracy
```
python results_evaluate.py
```
- Global Rejoining
```
python Global_Rejoin.py
```
-  Calculate parameters and MACs
```
MAC_params.py
```
### Dataset Creation
```
│data/
├──01_fragImge.py
│Through horizontal and vertical split curves, the bone stick images are divided into fragment images corresponding to top-bottom, bottom-top, left-right, and right-left rejoining.
├──02_dele_background.py
│Crop the segmented bone sticks, resizing the 1200×1170 image to 800×800 to remove a significant amount of background information.
├──03_remove_small_part_img.py
│Some fragments obtained after horizontal splitting have very short matching edges. Images with such short edges should not be used for top-bottom rejoining, so we need to remove them.
├──04_delet_small_dataset.py
│Remove records from the generated rejoinable txt document that contain images with too short matching edges (for both top-bottom and bottom-top rejoining).
```
- Dataset division
```
│data/
├──05_divide_dataset.py
│Divide the dataset into training, test, and validation sets
├──10_resize_train_val_test_imgs.py
│Resize all images in the dataset (train, test, and validation) to 224×224 size.
```
- Generate the txt file for the dataset
```
│data/
├──06_create_train_val_not_rejoin_txt.py
│Generate unrejoinable data for the train and validation set. 
├──07_train_val_txt.py
│Generate txt files for train and validation set
├──08_create_test_not_rejoin_full_txt.py
│Generate unrejoinable data for the test set
├──09_creat_test_full_txt.py
│Generate txt files for test set
```

- Dataset Structure
```
│data/
├──Train/
│  ├── 00001_1.bmp
│  ├── 00001_2.bmp
│  ├── ......
├──Val/
│  ├── 00048_1.bmp
│  ├── 00048_2.bmp
│  ├── ......
├──Test/
│  ├── 00002_1.bmp
│  ├── 00002_2.bmp
│  ├── ......
├──Train.txt
├──Val.txt
├──Test_full.txt
```

