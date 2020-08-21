import os
import torch
import numpy as np

import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO

from dataset.trasform import transform_COCO
from PIL import Image

import matplotlib
matplotlib.use('TkAgg')
from matplotlib.patches import Rectangle


class COCO_Dataset(Dataset):
    """Coco dataset."""

    def __init__(self, root_dir='D:\Data\coco', set_name='train2017', split='TRAIN'):

        """
        Args:
            root_dir (string): COCO directory.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        '''
        data path is as follos:
        root -- images      -- train2017
             |              |- val2017
             |              (|- test2017)
             | 
             -- anotations  -- instances_train2017.json
                            |- instances_val2017.json  * minival 
                            (|- image_info_test2017.json)
                            (|- image_info_test-dev2017.json)
        '''
        super().__init__()
        self.root_dir = root_dir
        self.set_name = set_name
        self.coco = COCO(os.path.join(self.root_dir, 'annotations', 'instances_' + self.set_name + '.json'))

        whole_image_ids = self.coco.getImgIds()  # original length of train2017 is 118287
        self.image_ids = []
        # to remove not annotated image idx
        self.no_anno_list = []
        for idx in whole_image_ids:
            annotations_ids = self.coco.getAnnIds(imgIds=idx, iscrowd=False)
            if len(annotations_ids) == 0:
                self.no_anno_list.append(idx)
            else:
                self.image_ids.append(idx)
        # after removing not annotated image, the length of train2017 is 117359
        # in https://github.com/cocodataset/cocoapi/issues/76 1021 not annotated images exist
        # so 118287 - 117266 = 1021

        self.load_classes()
        self.split = split

    def load_classes(self):
        # load class names (name -> label)
        categories = self.coco.loadCats(self.coco.getCatIds())
        categories.sort(key=lambda x: x['id'])

        self.classes = {}
        self.coco_labels = {}
        self.coco_labels_inverse = {}
        for c in categories:
            self.coco_labels[len(self.classes)] = c['id']
            self.coco_labels_inverse[c['id']] = len(self.classes)
            self.classes[c['name']] = len(self.classes)

        # also load the reverse (label -> name)
        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):

        visualize = False

        image, (w, h) = self.load_image(idx)
        annotation = self.load_annotations(idx)

        boxes = torch.FloatTensor(annotation[:, :4])
        labels = torch.LongTensor(annotation[:, 4])

        if labels.nelement() == 0:  # no labeled img exists.
            visualize = True
        # data augmentation
        image, boxes, labels = transform_COCO(image, boxes, labels, self.split)

        if visualize:
            # ----------------- visualization -----------------
            resized_img_size = 416

            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])

            # tensor to img
            img_vis = np.array(image.permute(1, 2, 0), np.float32)  # C, W, H
            img_vis *= std
            img_vis += mean
            img_vis = np.clip(img_vis, 0, 1)

            plt.figure('input')
            plt.imshow(img_vis)

            for i in range(len(boxes)):
                print(boxes[i], labels[i])
                plt.gca().add_patch(Rectangle((boxes[i][0] * resized_img_size, boxes[i][1] * resized_img_size),
                                              boxes[i][2] * resized_img_size - boxes[i][0] * resized_img_size,
                                              boxes[i][3] * resized_img_size - boxes[i][1] * resized_img_size,
                                              linewidth=1, edgecolor='r', facecolor='none'))

                plt.text(boxes[i][0] * resized_img_size - 5, boxes[i][1] * resized_img_size - 5,
                         str(self.labels[labels[i].item()]),
                         bbox=dict(boxstyle='round4', color='grey'))

            plt.show()

        return image, boxes, labels

    def load_image(self, image_index):
        image_info = self.coco.loadImgs(self.image_ids[image_index])[0]
        path = os.path.join(self.root_dir, 'images', self.set_name, image_info['file_name'])
        image = Image.open(path).convert('RGB')
        return image, (image_info['width'], image_info['height'])

    def load_annotations(self, image_index):
        # get ground truth annotations
        annotations_ids = self.coco.getAnnIds(imgIds=self.image_ids[image_index], iscrowd=False)
        annotations = np.zeros((0, 5))

        # some images appear to miss annotations (like image with id 257034)
        if len(annotations_ids) == 0:
            return annotations

        # parse annotations
        coco_annotations = self.coco.loadAnns(annotations_ids)
        for idx, a in enumerate(coco_annotations):

            # some annotations have basically no width / height, skip them
            if a['bbox'][2] < 1 or a['bbox'][3] < 1:
                continue

            annotation = np.zeros((1, 5))
            annotation[0, :4] = a['bbox']
            annotation[0, 4] = self.coco_label_to_label(a['category_id'])
            annotations = np.append(annotations, annotation, axis=0)

        # transform from [x, y, w, h] to [x1, y1, x2, y2]
        annotations[:, 2] = annotations[:, 0] + annotations[:, 2]
        annotations[:, 3] = annotations[:, 1] + annotations[:, 3]

        return annotations

    def coco_label_to_label(self, coco_label):
        return self.coco_labels_inverse[coco_label]

    def label_to_coco_label(self, label):
        return self.coco_labels[label]

    def image_aspect_ratio(self, image_index):
        image = self.coco.loadImgs(self.image_ids[image_index])[0]
        return float(image['width']) / float(image['height'])

    def num_classes(self):
        return 80

    def collate_fn(self, batch):
        """
        :param batch: an iterable of N sets from __getitem__()
        :return: a tensor of images, lists of varying-size tensors of bounding boxes, labels, difficulties, img_name and
        additional_info
        """
        images = list()
        boxes = list()
        labels = list()

        for b in batch:
            images.append(b[0])
            boxes.append(b[1])
            labels.append(b[2])

        images = torch.stack(images, dim=0)
        return images, boxes, labels


if __name__ == '__main__':
    train_set = COCO_Dataset()
    train_loader = DataLoader(train_set,
                              batch_size=1,
                              collate_fn=train_set.collate_fn,
                              shuffle=False,
                              num_workers=0,
                              pin_memory=True)

    for i, (images, boxes, labels) in enumerate(train_loader):
        images = images.cuda()
        boxes = [b.cuda() for b in boxes]
        labels = [l.cuda() for l in labels]



