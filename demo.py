from torchvision import transforms
from PIL import Image
from model import YOLO_VGG_16
import torch
import os
import glob
import cv2
import time
from utils import make_pred_bbox, voc_labels_array, device, color_array
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def save_det_txt_for_mAP(file_name, bbox, cls, score):
    # type : (string, Tensor, list, Tensor )
    '''
    to save filename.txt to use input of https://github.com/Cartucho/mAP python evaluation codes.
    :param file_name: file name
    :param bbox: bbox tensor [num_obj, 4]
    :param cls: class list [num_obj]
    :param score: score tensor [num_obj]
    :return: None
    '''

    # score = score[0] # not batch results
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
    # type : (PIL Image, nn.Module, float)
    """
    to demo detection output using our models
    :param original_image: input image for detecting
    :param model: our yolo v2 vgg 16
    :param conf_thres: above conf score, detector detect by scores.
    :return: det_boxes - detected bbox
             det_labels - detected labels
             scores - detected bbox's scores
             detection_time - detection time for inference
    """

    # Transform
    image = normalize(to_tensor(resize(original_image)))

    # Move to default device
    image = image.to(device)

    tic = time.time()

    # Forward prop. # make batch 1
    preds = model(image.unsqueeze(0))
    preds = preds.permute(0, 2, 3, 1).cpu()  # B, 13, 13, 125
    bbox, cls, scores = make_pred_bbox(preds=preds, conf_threshold=conf_thres)  # 가장큰 네모 1개 뽑기 not batch score

    # detection time
    detection_time = time.time() - tic

    # gpu to cpu
    det_boxes = bbox.to('cpu')

    # Transform to original image dimensions
    original_dims = torch.FloatTensor(
        [original_image.width, original_image.height, original_image.width, original_image.height]).unsqueeze(0)
    det_boxes = det_boxes * original_dims
    det_labels = cls

    return det_boxes, det_labels, scores, detection_time


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--demo_img_path', type=str, default='D:\Data\VOC_ROOT\TEST\VOC2007\JPEGImages')
    parser.add_argument('--demo_img_type', type=str, default='jpg')
    parser.set_defaults(visualization=True)
    parser.set_defaults(save_demo=True)
    parser.add_argument('--vis', dest='visualization', action='store_true')
    parser.add_argument('--epoch', type=int, default=149)
    parser.add_argument('--save_path', type=str, default='./saves')
    parser.add_argument('--save_file_name', type=str, default='yolo_v2_vgg_16')
    parser.add_argument('--conf_thres', type=float, default=0.35)
    demo_opts = parser.parse_args()
    print(demo_opts)

    visualization = demo_opts.visualization
    save_demo = demo_opts.save_demo
    epoch = demo_opts.epoch

    model = YOLO_VGG_16().to(device)
    checkpoint = torch.load(os.path.join(demo_opts.save_path, demo_opts.save_file_name) + '.{}.pth.tar'.format(epoch))
    model.load_state_dict(checkpoint['model_state_dict'])

    model = model.to(device)
    model.eval()

    # Transforms
    resize = transforms.Resize((416, 416))
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    # test root - put your demo folder and types
    img_path = demo_opts.demo_img_path
    img_paths = glob.glob(os.path.join(img_path, '*.' + demo_opts.demo_img_type))

    tic = time.time()
    total_time = 0

    print(len(img_paths))
    with torch.no_grad():
        for j, img_path in enumerate(img_paths):

            # for each a image, outputs are boxes and labels.
            img = Image.open(img_path, mode='r').convert('RGB')
            boxes, labels, scores, det_time = demo(img, model=model, conf_thres=demo_opts.conf_thres)

            name = os.path.basename(img_path).split('.')[0]  # .replace('.jpg', '.txt')
            # save_det_txt_for_mAP(file_name=name, bbox=boxes, cls=labels, score=scores)

            total_time += det_time
            if j % 100 == 0:
                print("[{}/{}]".format(j, len(img_paths)))
                print("fps : {:.4f}".format(j / total_time))

            bbox = boxes
            cls = labels

            if visualization:
                images = img

                # 2. RGB to BGR
                image_np = np.array(images)

                plt.figure('result_{}'.format(j))
                plt.imshow(image_np)

                for i in range(len(bbox)):
                    print(cls[i])

                    plt.text(x=bbox[i][0],
                             y=bbox[i][1],
                             s=voc_labels_array[int(cls[i].item())] + ' {:.2f}'.format(scores[i]),
                             fontsize=10,
                             bbox=dict(facecolor=color_array[int(cls[i])],
                                       alpha=0.5))

                    plt.gca().add_patch(Rectangle(xy=(bbox[i][0], bbox[i][1]),
                                                  width=bbox[i][2] - bbox[i][0],
                                                  height=bbox[i][3] - bbox[i][1],
                                                  linewidth=1,
                                                  edgecolor=color_array[int(cls[i])],
                                                  facecolor='none'))

                if save_demo:
                    os.makedirs('./test_demo', exist_ok=True)
                    plt.savefig('./test_demo/result_{}.jpg'.format(j))

                plt.show()

        print("fps : {:.4f}".format(len(img_paths) / total_time))




