# yolo v2 vgg16 pytorch

:octocat: re-implementation of yolo v2 detection using torchvision vgg16 bn model.

### Intoduction

A pytorch implementation of vgg16 version of yolo v2 described in **YOLO9000: Better, Faster, Stronger**  [paper](https://arxiv.org/abs/1612.08242) by Joseph Redmon, Ali Farhadi.
The goal of this repo. is to re-implement a famous one-stage object detection, yolo v2 using torchvision models. 

### Requirements

- Python 3.7
- Pytorch 1.5.0
- visdom
- numpy 
- cv2
- matplotlib

### Results

|methods        | Traning Dataset              | Testing Dataset | Resolution | mAP     | Fps |
|---------------|------------------------------|:---------------:|:----------:|:-------:|:---:|
|original papers| VOC2007 train + VOC2012 train|   VOC2007 test  |  416 x 416 |   76.8  | 67  |
|ours           | VOC2007 train + VOC2012 train|   VOC2007 test  |  416 x 416 | **77.3**| 19  |


### Implementation

- ##### Dataset

Firstly, you make a dataset file structure like bellow for voc train and test.

VOCtrainval needs to contain TRAIN file and VOCtest contain TEST file.  
```bash
root|-- TEST
        |-- VOC2007
            |-- Annotations
            |-- ImageSets
            |-- JPEGImages
            |-- SegmentationClass
            |-- SegmentationObject
    |-- TRAIN
        |-- VOC2007
            |-- Annotations
            |-- ImageSets
            |-- JPEGImages
            |-- SegmentationClass
            |-- SegmentationObject
        |-- VOC2012
            |-- Annotations
            |-- ImageSets
            |-- JPEGImages
            |-- SegmentationClass
            |-- SegmentationObject
```
to train, we used voc2007trainval + voc2012trainval dataset,

to test, we used voc2007test dataset

- ##### Model

Unlike the existing yolo v2, the backbone uses vgg instead of darknet 19, and the modules behind it have been modified a little bit.

![model](https://user-images.githubusercontent.com/18729104/87277514-6821f200-c51d-11ea-9558-f5ecb9aece02.JPG)

- ##### Loss

![losses](https://user-images.githubusercontent.com/18729104/87278340-2b56fa80-c51f-11ea-89a3-600835beca8f.JPG)


xy centor loss 

wh ratio loss 

confidence loss

no conf loss

classification loss 

whole loss is sum of those losses

- ##### Train

|        Epoches       | Learning rate |
|----------------------|:---------------:|
|         000-099      |      1e-4     |
|         100-149      |      1e-5     |

- ##### Evaluation

evaluation is a voc metric, mAP(iou>0.5) and exactly same to official python mAP code https://github.com/Cartucho/mAP

### Start Guide for Train / Test / Demo

- for training


