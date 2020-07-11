import os
import torch
import time
from utils import make_pred_bbox, voc_labels_array
from voc_eval import voc_eval
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# --- for test
from dataset.voc_dataset import VOC_Dataset
from loss import Yolo_Loss
from model import YOLO_VGG_16


def test(epoch, device, vis, test_loader, model, criterion, save_path, save_file_name, conf_thres, eval=False):

    # ---------- load ----------
    print('Validation of epoch [{}]'.format(epoch))
    model.eval()
    check_point = torch.load(os.path.join(save_path, save_file_name) + '.{}.pth.tar'.format(epoch))
    state_dict = check_point['model_state_dict']
    model.load_state_dict(state_dict)

    visualization = False
    det_additional = list()
    det_boxes = list()
    det_labels = list()
    det_scores = list()

    tic = time.time()
    with torch.no_grad():

        for idx, (images, boxes, labels, difficulties, additional_info) in enumerate(test_loader):
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
                bbox, cls, scores = make_pred_bbox(preds=preds, conf_threshold=conf_thres)  # 가장큰 네모 1개 뽑기

                # bbox 가 검출이 되지 않으면 그냥 넘어간다.
                if bbox.size(0) == 1:
                    # print(idx, " continues")
                    continue

                det_additional.append(additional_info[0])
                det_boxes.append(bbox)
                det_labels.append(cls)
                det_scores.append(scores)
                # print(bbox)

                if visualization:

                    # --- visualization ---
                    # 0. permute
                    images = images.cpu()
                    images = images.squeeze(0).permute(1, 2, 0)  # B, C, H, W --> H, W, C
                    # 1. un normalization
                    images *= torch.Tensor([0.229, 0.224, 0.225])
                    images += torch.Tensor([0.485, 0.456, 0.406])
                    # 2. RGB to BGR
                    image_np = images.numpy()
                    # image_np = image_np[:, :, ::-1]

                    bbox *= 416

                    # predicted bbox visualization
                    plt.figure('result')
                    plt.imshow(image_np)

                    for i in range(len(bbox)):
                        print(cls[i])
                        plt.text(x=bbox[i][0],
                                 y=bbox[i][1],
                                 s=voc_labels_array[int(cls[i].item())] + str(scores[i].item()),
                                 fontsize=10,
                                 bbox=dict(facecolor='red', alpha=0.5))

                        plt.gca().add_patch(Rectangle(xy=(bbox[i][0], bbox[i][1]),
                                                      width=bbox[i][2] - bbox[i][0],
                                                      height=bbox[i][3] - bbox[i][1],
                                                      linewidth=1, edgecolor='r', facecolor='none'))

                    plt.show()

            toc = time.time() - tic
            # ---------- print ----------
            # for each steps
            if idx % 100 == 0:
                print('Epoch: [{0}]\t'
                      'Step: [{1}/{2}]\t'
                      'Time : {time:.4f}\t'
                      .format(epoch,
                              idx, len(test_loader),
                              time=toc))

        mAP = voc_eval("D:\Data\VOC_ROOT\TEST\VOC2007\Annotations", det_additional, det_boxes, det_scores, det_labels)

        if vis is not None:
            # loss plot
            vis.line(X=torch.ones((1, 2)).cpu() * epoch,  # step
                     Y=torch.Tensor([loss, mAP]).unsqueeze(0).cpu(),
                     win='test_loss',
                     update='append',
                     opts=dict(xlabel='step',
                               ylabel='test',
                               title='test loss',
                               legend=['test Loss', 'mAP']))


if __name__ == "__main__":

    # 1. epoch
    epoch = 149
    # 2. device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 3. visdom
    vis = None

    # 4. data set
    test_set = VOC_Dataset(root="D:\Data\VOC_ROOT\TEST", split='TEST')
    test_loader = torch.utils.data.DataLoader(test_set,
                                              batch_size=1,
                                              collate_fn=test_set.collate_fn,
                                              shuffle=False)
    # 6. network
    model = YOLO_VGG_16().to(device)

    # 7. loss
    criterion = Yolo_Loss(num_classes=20)

    test(epoch=epoch,
         device=device,
         vis=vis,
         test_loader=test_loader,
         model=model,
         criterion=criterion,
         save_path='./saves',
         save_file_name='yolo_v2_vgg_16',
         eval=True,
         conf_thres=0.01)







