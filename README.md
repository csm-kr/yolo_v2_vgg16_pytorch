# yolo v2 vgg16 pytorch

re-implementation of yolo v2 detection using torchvision vgg16 bn model.

### Intoduction

:octocat:

A pytorch implementation of vgg16 version of yolo v2 described in **YOLO9000: Better, Faster, Stronger**  [paper](https://arxiv.org/abs/1612.08242) by Joseph Redmon, Ali Farhadi.

The goal of this repo. is to re-implement famous one-stage object detection using torchvision models. 

### Requirements

this repo. is 
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


```bash
root|-- dataset
        |-- __init__.py
        |-- dataset.py
    |-- model
        |-- __init__.py
        |-- model.py
    |-- learning
        |-- __init__.py
        |-- evaluator.py
        |-- optimizer.py
    |-- .gitignore
    |-- data
        |-- train.py
        |-- test.py
    |-- logs
    |-- save
    |-- train.py
    |-- test.py
    |-- utils.py
```


- ##### Model

- ##### Loss

- ##### Train

|        Epoches       | Learning rate |
|----------------------|:---------------:|
|         000-099      |      1e-4     |
|         100-149      |      1e-5     |


- ##### Evaluation

### experiments
2. 
- pretrained vgg 16 bn (torchvision.model)
- conv 5_3
- skip module
- extra, final detection module 

### Loss

- xy centor loss 

- wh ratio loss 

- confidence loss

- no conf loss

- classification loss 

- whole loss is sum of those losses

### Dataset

for train 
- voc 2007 train + voc 2012 train

for test
- voc 2007 test

### evaluation 

- mAP (metric for object detection using voc 2012 IOU:>0.5)

- 57.3% not cover original darknet19 yolo v2 

### training 

- learning rate : 1e-4

- optimizer : sgd

- lr decay : 0 ~ 99 (1e-4), 100~149(1e-5)

### experiments

1 -  classification cross entropy version 

- epoch 19 - 47.79% mAP
- epoch 20 - 49.05% mAP
- epoch 40 - 50.07% mAP

2 - new wh loss for (log)

- epoch 20 - 49.37% mAP

3 - multi-scale training + L1 loss for each 

- 000 ~ 149 1e-4 51.22% mAP (149 epoch)
- 150 ~ 169 1e-5 55.04% mAP (169 epoch)

multi-scale training is not stable, changing large scale from small scale (384 --> 608), it will be easy to exploding loss.

***** Fix Critical errors that uses 0.5 scale anchor to get predict xy as loss and make bbox *****

then get 65.46% mAP for 169 exp3 results. 

4 - 416 scale only training use original paper loss  

049 epoch 62.42% mAP 
099 epoch 72.04% mAP
149 epoch 77.03% mAP

- 000 ~ 099 1e-4 
- 100 ~ 149 1e-5 

### Start Guide for Train / Test / Demo

- for training


