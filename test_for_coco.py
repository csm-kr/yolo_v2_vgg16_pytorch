import os
import torch
import time
from utils import make_pred_bbox, voc_labels_array, make_pred_bbox_for_COCO
from voc_eval import voc_eval
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import argparse
# --- for test
import json
from dataset.voc_dataset import VOC_Dataset
from dataset.coco_dataset import COCO_Dataset
from loss import Yolo_Loss
from pycocotools.cocoeval import COCOeval
from model import YOLO_VGG_16
import tempfile


def test(epoch, device, vis, test_loader, model, criterion, save_path, save_file_name, conf_thres, eval=False):
    # ---------- load ----------
    print('Validation of epoch [{}]'.format(epoch))
    model.eval()
    check_point = torch.load(os.path.join(save_path, save_file_name) + '.{}.pth.tar'.format(epoch))
    state_dict = check_point['model_state_dict']
    model.load_state_dict(state_dict, strict=True)

    with torch.no_grad():

        # for COCO evaluation
        results = []
        image_ids = []

        for idx, datas in enumerate(test_loader):
            '''
            + VOC dataset
            for VOC datasets, datas including follows:
            (images, boxes, labels, difficulties, img_names, additional_info)

            + COCO dataset
            but COCO dataset, (images, boxes, labels)
            '''
            images = datas[0]
            boxes = datas[1]
            labels = datas[2]

            # ---------- cuda ----------
            images = images.to(device)
            boxes = [b.to(device) for b in boxes]
            labels = [l.to(device) for l in labels]

            # ---------- loss ----------
            preds = model(images)
            preds = preds.permute(0, 2, 3, 1)  # B, 13, 13, 125

            loss, _ = criterion(preds, boxes, labels)
            # ---------- eval ----------
            if eval:
                # bbox, cls, scores = make_pred_bbox(preds=preds, conf_threshold=conf_thres)

                # coco
                boxes, classes, scores = make_pred_bbox_for_COCO(preds, conf_threshold=conf_thres)  # 클래수갯수가 다름
                boxes = boxes.cpu()
                classes = classes.cpu()
                scores = scores.cpu()

                visualization = False
                if visualization:
                    # 0. permute
                    images = images.cpu()
                    images = images.squeeze(0).permute(1, 2, 0)  # B, C, H, W --> H, W, C

                    # 1. un normalization
                    images *= torch.Tensor([0.229, 0.224, 0.225])
                    images += torch.Tensor([0.485, 0.456, 0.406])

                    # 2. RGB to BGR
                    image_np = images.numpy()

                    # 3. box scaling
                    # bbox *= 416

                    plt.figure('result')
                    plt.imshow(image_np)

                    for i in range(len(boxes)):
                        print(classes[i])
                        if int(classes[i]) == 79 and scores[i] == 0:
                            plt.text(x=boxes[i][0] * 416,
                                     y=boxes[i][1] * 416,
                                     s='background' + str(scores[i].item()),
                                     fontsize=10,
                                     bbox=dict(facecolor='red', alpha=0.5))

                        else:
                            plt.text(x=boxes[i][0] * 416,
                                     y=boxes[i][1] * 416,
                                     s=test_loader.dataset.labels[int(classes[i].item())] + str(scores[i].item()),
                                     fontsize=10,
                                     bbox=dict(facecolor='red', alpha=0.5))

                        plt.gca().add_patch(Rectangle(xy=(boxes[i][0] * 416, boxes[i][1] * 416),
                                                      width=boxes[i][2] * 416 - boxes[i][0] * 416,
                                                      height=boxes[i][3] * 416 - boxes[i][1] * 416,
                                                      linewidth=1, edgecolor='r', facecolor='none'))

                    plt.show()

                # step 0
                # coco eval parameter 2개 + image_ids 만들기
                image_id = test_loader.dataset.image_ids[idx]
                image_ids.append(image_id)

                # step 1
                # bbox x1, y1, x2, y2 to x1, y1, w, h coordinate 로 다시 바꿔줌
                boxes[:, 2] -= boxes[:, 0]
                boxes[:, 3] -= boxes[:, 1]

                image_info = test_loader.dataset.coco.loadImgs(image_id)[0]
                w = image_info['width']
                h = image_info['height']

                # re-scaling
                boxes[:, 0] *= w
                boxes[:, 2] *= w
                boxes[:, 1] *= h
                boxes[:, 3] *= h

                for pred_box, pred_label, pred_score in zip(boxes, classes, scores):  # prediction 한 object 의 갯수
                    coco_result = {
                        'image_id'    : image_id,
                        'category_id' : test_loader.dataset.label_to_coco_label(int(pred_label)),
                        'score'       : float(pred_score),
                        'bbox'        : pred_box.tolist(),
                    }
                    results.append(coco_result)
                if idx % 1000 == 0:
                    print('{}/{}'.format(idx, test_loader.dataset.__len__()))
                # print('{}/{}'.format(idx, test_loader.dataset.__len__()), end='\r')  # end='\r' 을 사용하면 애니메이션처럼!

        # if not len(results):
        #     return
        _, tmp = tempfile.mkstemp()
        json.dump(results, open(tmp, 'w'))

        # json.dump(results, open('{}_bbox_results.json'.format(test_loader.dataset.set_name), 'w'))
        cocoGt = test_loader.dataset.coco
        cocoDt = cocoGt.loadRes(tmp)
        # https://github.com/argusswift/YOLOv4-pytorch/blob/master/eval/cocoapi_evaluator.py
        # workaround: temporarily write data to json file because pycocotools can't process dict in py36.

        coco_eval = COCOeval(cocoGt=cocoGt, cocoDt=cocoDt, iouType='bbox')
        coco_eval.params.imgIds = image_ids
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        ap = coco_eval.stats

        return


if __name__ == "__main__":
    # 1. argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_epoch', type=int, default=0)
    parser.add_argument('--save_path', type=str, default='./saves')
    parser.add_argument('--save_file_name', type=str, default='yolo_v2_vgg_16_coco')
    parser.add_argument('--conf_thres', type=float, default=0.1)
    from config import device
    test_opts = parser.parse_args()
    print(test_opts)

    epoch = test_opts.test_epoch

    # 2. device
    device = device

    # 3. visdom
    vis = None

    # 4. data set
    test_set = COCO_Dataset(set_name='val2017', split='TEST')
    test_loader = torch.utils.data.DataLoader(test_set,
                                              batch_size=1,
                                              collate_fn=test_set.collate_fn,
                                              shuffle=False)
    # 6. network
    model = YOLO_VGG_16(num_classes=80).to(device)

    # 7. loss
    criterion = Yolo_Loss(num_classes=80)

    test(epoch=epoch,
         device=device,
         vis=vis,
         test_loader=test_loader,
         model=model,
         criterion=criterion,
         save_path=test_opts.save_path,
         save_file_name=test_opts.save_file_name,
         eval=True,
         conf_thres=test_opts.conf_thres)







