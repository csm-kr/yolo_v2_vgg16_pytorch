import os
import json
import glob
import shutil
import xml.etree.ElementTree as ET
import torch
import time
import numpy as np


def parse_voc(filename):
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text
        obj_struct['difficult'] = int(obj.find('difficult').text)
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [int(bbox.find('xmin').text),
                              int(bbox.find('ymin').text),
                              int(bbox.find('xmax').text),
                              int(bbox.find('ymax').text)]
        objects.append(obj_struct)

    return objects


def save_gt(xml_file, cache_dir, classes, gt_counter_per_class):
    """

    :param xml_file:
    :param temp_dir:
    :param classes: class
    :return:
    """
    objects = parse_voc(xml_file)
    gt_name = os.path.basename(xml_file).split('.')[0]  # 이름만 뽑아오는 부분
    obj_dicts = []                                      # json 에 들어갈 내용
    for obj in objects:
        class_name = obj['name']
        bbox = str(obj['bbox'][0]) + " " + str(obj['bbox'][1]) + " " + str(obj['bbox'][2]) + " " + str(obj['bbox'][3])

        # parsing diff is 0
        if obj['difficult']:
            difficult = True
        else:
            difficult = False
            if class_name not in classes:
                classes.append(class_name)

            if class_name in gt_counter_per_class:
                gt_counter_per_class[class_name] += 1
            else:
                gt_counter_per_class[class_name] = 1

        obj_dicts.append({"class_name": class_name, "bbox": bbox, "used": False, "difficult": difficult})

    # .json 저장하는 부분
    new_temp_file = cache_dir + "/" + gt_name + "_ground_truth.json"
    with open(new_temp_file, 'w') as outfile:
        json.dump(obj_dicts, outfile)
    return classes, gt_counter_per_class


def save_pred(additional, bboxes, scores, classes, class_name, gt_classes, cache_dir):
    """

    :param add: img_name, img_width, img_height
    :param box:
    :param score:
    :param class_:
    :return:
    """

    preds_dicts = []
    for (add, obj_boxes, obj_scores, obj_class) in zip(additional, bboxes, scores, classes):
        img_name = add[0]
        img_width = add[1]
        img_height = add[2]

        # 1. 이름을 string 으로 바꾸어라
        img_name = str(int(img_name.item())).zfill(6)

        # 2. width, height 로 bbox 를 변형하라
        origin_wh = torch.FloatTensor([img_width, img_height, img_width, img_height]).unsqueeze(0)  # [1  , 4]
        obj_boxes = obj_boxes * origin_wh                                                           # [obj, 4]

        for (box, score, class_) in zip(obj_boxes, obj_scores, obj_class):
            # obj 개
            bbox = str(box[0].item()) + " " + str(box[1].item()) + " " + str(box[2].item()) + " " + str(box[3].item())
            confidence = str(score.item())  # [1]

            class_num = int(class_.item())  # for background
            if class_num == 20:
                continue

            if class_name == gt_classes[int(class_.item())]:
                preds_dicts.append({"confidence": confidence, "file_id": img_name, "bbox": bbox})

        # .json 저장하는 부분
    preds_dicts.sort(key=lambda x: float(x['confidence']), reverse=True)
    new_temp_file = cache_dir + "/" + class_name + "_dr.json"
    with open(new_temp_file, 'w') as outfile:
        json.dump(preds_dicts, outfile)


def voc_ap(rec, prec):

    rec.insert(0, 0.0)     # insert 0.0 at begining of list
    rec.append(1.0)        # insert 1.0 at end of list
    mrec = rec[:]
    prec.insert(0, 0.0)    # insert 0.0 at begining of list
    prec.append(0.0)       # insert 0.0 at end of list
    mpre = prec[:]

    for i in range(len(mpre)-2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i+1])

    i_list = []
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i-1]:
            i_list.append(i) # if it was matlab would be i + 1

    ap = 0.0
    for i in i_list:
        ap += ((mrec[i]-mrec[i-1])*mpre[i])
    return ap, mrec, mpre


def cal_mAP(cache_dir, gt_classes, gt_counter_per_class, MINOVERLAP=0.5):

    sum_AP = 0.0
    count_true_positives = {}
    for class_index, class_name in enumerate(gt_classes):

        count_true_positives[class_name] = 0
        """
         Load detection-results of that class
        """
        dr_file = cache_dir + "/" + class_name + "_dr.json"
        dr_data = json.load(open(dr_file))

        """
         Assign detection-results to ground-truth objects
        """
        nd = len(dr_data)
        tp = [0] * nd  # creates an array of zeros of size nd
        fp = [0] * nd
        for idx, detection in enumerate(dr_data):
            file_id = detection["file_id"]

            gt_file = cache_dir + "/" + file_id + "_ground_truth.json"
            ground_truth_data = json.load(open(gt_file))

            ovmax = -1
            gt_match = -1
            # load detected object bounding-box
            bb = [float(x) for x in detection["bbox"].split()]
            for obj in ground_truth_data:
                # look for a class_name match
                if obj["class_name"] == class_name:
                    bbgt = [float(x) for x in obj["bbox"].split()]
                    bi = [max(bb[0], bbgt[0]), max(bb[1], bbgt[1]), min(bb[2], bbgt[2]), min(bb[3], bbgt[3])]
                    iw = bi[2] - bi[0] + 1
                    ih = bi[3] - bi[1] + 1
                    if iw > 0 and ih > 0:
                        # compute overlap (IoU) = area of intersection / area of union
                        ua = (bb[2] - bb[0] + 1) * (bb[3] - bb[1] + 1) + (bbgt[2] - bbgt[0]
                                                                          + 1) * (bbgt[3] - bbgt[1] + 1) - iw * ih
                        ov = iw * ih / ua
                        if ov > ovmax:
                            ovmax = ov
                            gt_match = obj

            # set minimum overlap
            min_overlap = MINOVERLAP
            if ovmax >= min_overlap:
                if not gt_match["difficult"]:  # false difficult
                    if not bool(gt_match["used"]):
                        # true positive
                        tp[idx] = 1
                        gt_match["used"] = True
                        count_true_positives[class_name] += 1
                        # update the ".json" file
                        with open(gt_file, 'w') as f:
                            f.write(json.dumps(ground_truth_data))  # "used" 바꾸는 부분
                    else:
                        fp[idx] = 1  # false positive
            else:
                fp[idx] = 1          # false positive

        # compute precision/recall
        cumsum = 0
        for idx, val in enumerate(fp):
            fp[idx] += cumsum
            cumsum += val
        cumsum = 0
        for idx, val in enumerate(tp):
            tp[idx] += cumsum
            cumsum += val

        rec = tp[:]
        for idx, val in enumerate(tp):
            rec[idx] = float(tp[idx]) / gt_counter_per_class[class_name]

        prec = tp[:]
        for idx, val in enumerate(tp):
            # if (fp[idx] + tp[idx]) == 0:
            #     prec[idx] = 0
            # avoid divide by zero in case the first detection matches a difficult
            prec[idx] = float(tp[idx]) / np.maximum((fp[idx] + tp[idx]), np.finfo(np.float64).eps)

        ap, mrec, mprec = voc_ap(rec[:], prec[:])
        sum_AP += ap
        print("{0:.2f}%".format(ap * 100) + " = " + class_name + " AP ")

    mAP = sum_AP / len(gt_classes)
    print("mAP = {0:.2f}%".format(mAP * 100))
    return mAP


def voc_eval(test_xml_path="D:\Data\VOC_ROOT\TEST\VOC2007\Annotations",
             additional=None, bboxes=None, scores=None, classes=None):
    """

    :param test_xml_path: annotation path
    :return:
    """
    print("start..evaluation")
    tic = time.time()
    if bboxes is not None:
        # 0. cpu to cpu
        bboxes = [b.cpu() for b in bboxes]
        scores = [s.cpu() for s in scores]
        classes = [c.cpu() for c in classes]

    cache_dir = '.cache_dir'
    if not os.path.isdir(cache_dir):
        os.mkdir(cache_dir)

    gt_classes = []
    gt_counter_per_class = {}
    # 1. save gt
    xml_list = glob.glob(os.path.join(test_xml_path, '*.xml'))
    for xml_file in xml_list:
        gt_classes, gt_counter_per_class = save_gt(xml_file, cache_dir, gt_classes, gt_counter_per_class)

    # 2. get whole classes
    gt_classes = sorted(gt_classes)

    # 3. save_pred
    for class_index, class_name in enumerate(gt_classes):
        save_pred(additional, bboxes, scores, classes, class_name, gt_classes, cache_dir)

    # 4. calculate mAP
    map = cal_mAP(cache_dir, gt_classes, gt_counter_per_class, 0.5)
    shutil.rmtree(cache_dir, ignore_errors=True)

    toc = time.time() - tic
    print("it takes {:.2f}sec".format(toc))

    return map


if __name__ == '__main__':
    voc_eval(test_xml_path="D:\Data\VOC_ROOT\TEST\VOC2007\Annotations")