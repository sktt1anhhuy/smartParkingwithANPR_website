# smartParkingwithANPR_website

## Installation Guide
1. Install [python](https://www.python.org/downloads/)
2. Install [git](https://git-scm.com/)
3. Install [PyCharm](https://www.jetbrains.com/pycharm/)
4. Create a project in PyCharm
5. At terminal of project in Pycharm, type 
  ```python
git clone https://github.com/sktt1anhhuy/smartParkingwithANPR_website.git
```
6. Install essential library
```python
pip install -r requirements.txt
```
7. Run the web app, type at terminal
```python
cd smartParkingwithANPR_website
streamlit run [your directory to]/app.py # for example E:/smartParkingwithANPR_website
```
## Result

https://github.com/sktt1anhhuy/smartParkingwithANPR_website/blob/master/video_demo.webm

# Quick Guide

## Data
The data is used for training Licence Plate Detection is the combination from this [link](https://github.com/thigiacmaytinh/DataThiGiacMayTinh/blob/main/GreenParking.zip) for only one Motorbike Licence Plate and from internet by searching with some specific keyword such as "2 biển số xe máy", "nhiều xe máy". Then this data is labeled by using [CVAT](https://www.cvat.ai/) and download in Yolov1.1 format

## Training
1.  In "smartParkingwithANPR_website" directory, create a folder name "data"
  - In folder "data", create two subfolders named "images" and "labels"
  - In folder "images" create three subfolders named "test", "train", and "val"
  - In folder "labels" create three subfolders named "test", "train", and "val"
  - In put your images data into three subfolder "test", "train", and "val" of folder "images"
  - In put your labels data into three subfolder "test", "train", and "val" of folder "labels"
2. In file data.yaml
```python
path: [you project disk]/smartParkingwithANPR_website/data # Example: D:/smartParkingwithANPR_website/data
train: images/train
val: images/val
test: images/test

# You can increase the class object corresponding to your lable data
# Classes
names:
  0: motorbike plate
  # 1: car plate
  # 2: bird
  # 3: duck
```
3. Run the file train.py
4. After traning the weights of the model will be save at "runs/detect/train" in smartParkingwithANPR_website named "best.pt". Remember to change the name "best.pt" to "best1.pt" or some name you want to avoid duplication since we already have a file name "best.pt". You run the "app.py" with your own weight by changing the variable name line 38 in "app.py" to yours weight name
```python
model_path = 'best.pt' # Example: change to model_path = 'best1.pt'
```


