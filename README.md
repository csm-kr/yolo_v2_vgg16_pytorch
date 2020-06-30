html header: <script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>

# YOLO V2 detection using vgg16

re-implementation of yolo v2 detection 

### Setting

- Python 3.7
- Pytorch 1.5.0
- visdom
- numpy 

### Network

- pretrained vgg 16 bn (torchvision.model)

- conv 5_3

- skip module

- extra, final detection module 

### Loss

- xy centor loss 
$$\sum_i\sum_j$$

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

1. 

2. 

### Start Guide

- for training

### Result 

- qualitative results

- qualified results

