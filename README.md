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

![result](https://user-images.githubusercontent.com/18729104/87284497-65c99300-c531-11ea-881d-e4ab1917b49a.gif)

detection result of voc 2007. 

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

![model](https://user-images.githubusercontent.com/18729104/87281786-8a703b80-c52e-11ea-9f1f-3bb1d3d23a58.JPG)
- ##### Loss

*01- what is the cell concept?*

yolo considers the final layer feature map size as a cell size. 

For example, an image of 416 resolution becomes a cell of 13 size.

*02- make_target*

to assign gt bbox to anchors. 

so we get positive anchors if iou(bbox, anchors) > 0.5

For positive anchors, it is xy_gt that scales 0 to 1 to which position in the center of gt_bbox corresponds to the cell.

also, for positive anchors, wh_gt is the ratio of gt_bbox and anchor boxes.

gt_conf is max iou(pred_bbox, gt_bbox) for each anchor in cells. 

no conf is 1 - gt_conf 

*03- whole loss*

whole loss consists of xy centor loss, wh ratio loss, confidence loss, no conf loss, and classification loss.
original paper losses are sum square errors of each component, except to wh ration loss is root sse. 

![yolo_v2_losses](https://user-images.githubusercontent.com/18729104/87280599-4af51f80-c52d-11ea-86c7-f4dc8786f827.JPG)

- ##### Train

optimizer is SGD (weight_decay : 5e-4, momentum : 0.9)

train until convergence (about 150 epochs)

learning rate decay

|        Epoches       | Learning rate |
|----------------------|:-------------:|
|         000-099      |      1e-4     |
|         100-149      |      1e-5     |

- ##### Evaluation

evaluation is a voc metric, mAP(iou>0.5) and exactly same to official python mAP code https://github.com/Cartucho/mAP

### Start Guide for Train / Test / Demo

- for training

```
# python main.py 
usage: main.py [-h] [--batch_size] [--conf_thres] 
               [--epochs] [--lr] [--num_workers]
               [--save_file_name] [--save_path] [--start_epoch]

  -h, --help            show this help message and exit
  --batch_size          for training batch size, test batch is only 1.
  --conf_thres          for testing, confience threshold, default 0.01 
  --epochs              whole traning epochs 
  --lr                  initial learning rate (default 1e-3) 
  --num_workers         dataset num_workers (default 2)
  --save_file_name      when you do experiment, you can change save_file_name to distinguish other pths.
  --save_path           the path to save .pth file
  --start_epoch         when you resume, set the start epochs. 
```

before test and demo, we must have trained .pth file(weight params) you can download 149 epoch weights
at https://drive.google.com/file/d/1cOadYYkwXKzAf7YGYokEAEn1ofzx4QDX/view?usp=sharing

and then make ./saves file place the weight in the file.

- for testing

```
# python test.py 
usage: test.py [-h] [--test_epoch] [--save_path] 
               [--save_file_name] [--conf_thres]

  -h, --help            show this help message and exit
  --test_epoch          for testing, which epoch param do we get
  --save_path           for testing, params path (default './saves') 
  --save_file_name      save_file_name to distinguish other params. (default 'yolo_v2_vgg_16')
  --conf_thres          for testing, confience threshold which detector detect above the thres (default 0.01) 
```

- for demo

```
# python demo.py 
usage: demo.py [-h] [--demo_img_path] [--demo_img_type] 
               [--visualization] [--epoch] [--save_path]
               [--save_file_name] [--conf_thres]

  -h, --help            show this help message and exit
  --demo_img_path       The path that contains the image you want to detect
  --demo_img_type       The type of images you want to detect
  --visualization       Whether to visualize (default True)
  --epoch               for demo, which epoch param do we get
  --save_path           for demo, params path (default './saves') 
  --save_file_name      save_file_name to distinguish other params. (default 'yolo_v2_vgg_16')
  --conf_thres          for demo, confience threshold which detector detect above the thres (default 0.35) 
```
