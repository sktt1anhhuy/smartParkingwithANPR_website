# smartParkingwithANPR_website

## Installation Guilde
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
# Quick Guild

## Data
The data is used for training Licence Plate Detection is the combination from this [link](https://github.com/thigiacmaytinh/DataThiGiacMayTinh/blob/main/GreenParking.zip) for only one Motorbike Licence Plate and from internet by searching with some specific keyword such as "2 biển số xe máy", "nhiều xe máy". Then this data is labeled by using [CVAT](https://www.cvat.ai/) and download in Yolov1.1 format

## Training

1. Run the file train.py
