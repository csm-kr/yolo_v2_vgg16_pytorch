from torchvision import transforms
from PIL import Image
from model import YOLO_VGG_16
import torch
import os
import glob
import cv2
import time
from utils import make_pred_bbox, voc_labels_array, device


def save_det_txt_for_mAP(file_name, bbox, cls, score):
    '''
    file name 을 mAP 에 넣을 수 있도록 만드는 부분
    :param file_name:
    :param bbox:
    :param cls:
    :param score:
    :return:
    '''

    # score = score[0]
    if not os.path.isdir('./pred'):
        os.mkdir('./pred')
    f = open(os.path.join("./pred", file_name + '.txt'), 'w')
    for idx, t in enumerate(bbox):
        if cls[idx] == 'background':
            continue
        class_name = cls[idx]
        data = class_name + \
               " " + str(score[idx].item()) + \
               " " + str(t[0].item()) + \
               " " + str(t[1].item()) + \
               " " + str(t[2].item()) + \
               " " + str(t[3].item()) + "\n"
        f.write(data)
    f.close()


def demo(original_image, model, conf_thres):
    """

    :param original_image:
    :param model:
    :param conf_thres:
    :return:
    """

    # Transform
    image = normalize(to_tensor(resize(original_image)))

    # Move to default device
    image = image.to(device)

    tic = time.time()

    # Forward prop. # make batch 1
    preds = model(image.unsqueeze(0))
    preds = preds.permute(0, 2, 3, 1)  # B, 13, 13, 125
    bbox, cls, scores = make_pred_bbox(preds=preds, conf_threshold=conf_thres)  # 가장큰 네모 1개 뽑기

    # detection time
    detection_time = time.time() - tic

    # gpu to cpu
    det_boxes = bbox.to('cpu')

    # Transform to original image dimensions
    original_dims = torch.FloatTensor(
        [original_image.width, original_image.height, original_image.width, original_image.height]).unsqueeze(0)
    det_boxes = det_boxes * original_dims
    det_labels = [voc_labels_array[l] for l in cls.to('cpu').tolist()]

    return det_boxes, det_labels, scores, detection_time


if __name__ == '__main__':

    visualization = False
    epoch = 149

    model = YOLO_VGG_16().to(device)
    checkpoint = torch.load(os.path.join('./saves', 'yolo_v2_vgg_16') + '.{}.pth.tar'.format(epoch))
    model.load_state_dict(checkpoint['model_state_dict'])

    model = model.to(device)
    model.eval()

    # Transforms
    resize = transforms.Resize((416, 416))
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    # test root - put your demo folder and types
    img_path = 'D:\Data\VOC_ROOT\TEST\VOC2007\JPEGImages'
    img_paths = glob.glob(os.path.join(img_path, '*.jpg'))

    tic = time.time()
    total_time = 0

    print(len(img_paths))
    with torch.no_grad():
        for i, img_path in enumerate(img_paths):

            # for each a image, outputs are boxes and labels.
            img = Image.open(img_path, mode='r').convert('RGB')
            boxes, labels, scores, det_time = demo(img, model=model, conf_thres=0.01)

            name = os.path.basename(img_path).split('.')[0]  # .replace('.jpg', '.txt')
            save_det_txt_for_mAP(file_name=name, bbox=boxes, cls=labels, score=scores)

            total_time += det_time
            if i % 100 == 0:
                print("[{}/{}]".format(i, len(img_paths)))
                print("fps : {:.4f}".format(i / total_time))

            if visualization:
                img = cv2.imread(img_path)
                scores = scores[0]  # score is list of tensors
                for i in range(len(boxes)):
                    cv2.rectangle(img,
                                  pt1=(boxes[i][0], boxes[i][1]),
                                  pt2=(boxes[i][2], boxes[i][3]),
                                  color=(0, 0, 255),
                                  thickness=2)

                    cv2.putText(img,
                                text=labels[i],
                                org=(boxes[i][0] + 10, boxes[i][1] + 10),
                                fontFace=0, fontScale=0.7,
                                color=(255, 255, 0))

                cv2.imshow('input', img)
                cv2.waitKey(0)

        print("fps : {:.4f}".format(len(img_paths) / total_time))




