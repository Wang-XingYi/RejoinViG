# RejoinViG

# Overview
This repository contains the source code for Rejoining Precious Artifacts: Efficiently Bone Stick Rejoining Based Massive Fragment Images by Contour, Script, and Texture
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
python test.py
```

<details>
<summary>Calculate Top-K accuracy </summary>

```
python tools/results_evaluate.py
```

</details>

<details>
<summary>Global rejoining </summary>

```
python tools/Global_Rejoin.py
```

</details>

<details>
<summary>Calculate parameters and MACs</summary>

```
python tools/MAC_params.py
```
</details>

### 📁Prepare Dataset
<details>
<summary>Dataset Creation </summary>

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
</details>


<details>
<summary>Dataset division </summary>
```
│data/
├──05_divide_dataset.py
│Divide the dataset into training, test, and validation sets
├──10_resize_train_val_test_imgs.py
│Resize all images in the dataset (train, test, and validation) to 224×224 size.
```
</details>


<details>
<summary>Generate the txt file for the dataset </summary>

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

</details>

<details>
<summary>Dataset structure </summary>

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
</details>

## Acknowledgement
Our work is built upon [GreedyViG](https://github.com/SLDGroup/GreedyViG).
Thanks to the inspirations from [GreedyViG](https://github.com/SLDGroup/GreedyViG).

✨ Feel free to contribute and reach out if you have any questions! ✨
