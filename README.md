
# Indoor Object Detection with YOLOv8
This project implements an object detection system for indoor scenes using the YOLOv8 model. It is designed to detect various objects in indoor environments such as doors, cabinets, tables, chairs, and more. The dataset used for training is from the Indoor Object Detection Kaggle Dataset.

## Project Overview
This project trains a YOLOv8 model on a custom indoor object detection dataset and uses the trained model for object detection in images and real-time webcam feed.





```
├── YOLO/
  ├── dataset/
  │   │   ├── train/
  │   │   └── test/
  │   │   ├── valid/
  │   ├── README.md
  │   ├── data.yaml
  ├── gputest.py 
  ├── train.py                # Training script for YOLOv8
  ├── test.py               # Object detection script using webcam
  ├── test.jpg
  ├── requirements.txt        # List of required libraries
```
## requirements
To get started, you need to install the following dependencies. You can do so by running:
```
pip install -r requirements.txt
```
### dependencies:

* ultralytics
* opencv-python
* torch
* matplotlib
* numpy

## Installation

1. Clone the repository:
```
git clone https://github.com/yourusername/indoor-object-detection.git
cd indoor-object-detection

```
2. Install the required dependencies:
```
pip install -r requirements.txt

```
## dataset
The dataset used for this project is the [kaggle](https://www.kaggle.com/datasets/thepbordin/indoor-object-detection), which contains labeled images of various indoor objects. The dataset is divided into:
* `train`: Training images
* `test`: Test images
* `labels`: Corresponding annotation files for the images
#
### data.yaml File
The `data.yaml` file contains the path to the dataset and the classes being detected. It should look like this:
```
train: dataset/images/train
val: dataset/images/test
nc: 10
names: ['door', 'cabinetDoor', 'refrigeratorDoor', 'window', 'chair', 'table', 'cabinet', 'couch', 'openedDoor', 'pole']

```
## Training the Model
To train the model on your dataset, use the following command:
```
python train.py
```
This will train the model and save the best weights to 
`runs/detect/train11/weights/best.pt`.

# Problem in GPU or CUDA

if u face any problem Regarding CUDA or GPU

Uninstall Existing PyTorch
```
pip uninstall torch torchvision torchaudio -y
```
Install PyTorch with CUDA (for RTX 3050 the gpu i used)
For CUDA 11.8 (Recommended)
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
## Object Detection

After training, you can use the trained model for object detection on test images or live webcam feed.

To run the detection script on an image, use the following command:

```
python detect.py --img_path "path_to_image.jpg"
```
For real-time webcam object detection, use:
```
python detect.py --webcam

```
## Project Description

`train.py`
This script is used to train the YOLOv8 model on the custom dataset. It loads the dataset, configures the training settings, and begins the training process.

`detect.py`
This script is used for performing object detection using the trained model. It can be used on static images or in real-time with the webcam.

`best.pt`
The model weights file that contains the best trained model. It is used to perform object detection.
```
the path for `best.pt` usually present at C:\users\youruser\runs\detect\train11\weights
```
## Pull something

`feel free to pull this repo and optimize the program`
