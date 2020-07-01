import torch.nn as nn
import torch.nn.functional as F
import torch
from anchor import make_center_anchors
from utils import find_jaccard_overlap, center_to_corner, corner_to_center


class Yolo_Loss(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.anchors = [(1.3221, 1.73145), (3.19275, 4.00944), (5.05587, 8.09892), (9.47112, 4.84053),
                        (11.2364, 10.0071)]

    def make_target(self, gt_boxes, gt_labels, pred_xy, pred_wh):
        """
        make
        :param gt_bboxes:
        :param gt_labels:
        :param pred_xy:
        :param pred_wh:
        :param anchors:
        :return:
        """
        out_size = pred_xy.size(2)
        # print("out_size :", out_size)

        batch_size = pred_xy.size(0)
        resp_mask = torch.zeros([batch_size, out_size, out_size, 5])  # y, x, anchor, ~
        gt_xy = torch.zeros([batch_size, out_size, out_size, 5, 2])
        gt_wh = torch.zeros([batch_size, out_size, out_size, 5, 2])
        gt_conf = torch.zeros([batch_size, out_size, out_size, 5])
        gt_cls = torch.zeros([batch_size, out_size, out_size, 5, 20])

        center_anchors = make_center_anchors(anchors_wh=self.anchors, grid_size=out_size)
        corner_anchors = center_to_corner(center_anchors).view(out_size * out_size * 5, 4)

        # 1. make resp_mask
        for b in range(batch_size):

            label = gt_labels[b]
            corner_gt_box = gt_boxes[b]
            corner_gt_box_13 = corner_gt_box * float(out_size)

            center_gt_box = corner_to_center(corner_gt_box)
            center_gt_box_13 = center_gt_box * float(out_size)

            bxby = center_gt_box_13[..., :2]  # [# obj, 2]
            txty = bxby - bxby.floor()        # [# obj, 2], 0~1 scale

            bwbh = center_gt_box_13[..., 2:]

            iou_anchors_gt = find_jaccard_overlap(corner_anchors, corner_gt_box_13)  # [845, # obj]
            iou_anchors_gt = iou_anchors_gt.view(out_size, out_size, 5, -1)

            num_obj = corner_gt_box.size(0)
            for n_obj in range(num_obj):
                cx, cy = bxby[n_obj]
                cx = int(cx)
                cy = int(cy)

                _, max_idx = iou_anchors_gt[cy, cx, :, n_obj].max(0)  # 어떤 anchor 에서 maximum 을 갖느냐?
                j = max_idx  # 얘는 idx 이다.
                # # j 번째 anchor
                resp_mask[b, cy, cx, j] = 1
                gt_xy[b, cy, cx, j, :] = txty[n_obj]

                twth = bwbh[n_obj] / torch.Tensor(self.anchors[j]).cuda()   # 비율
                gt_wh[b, cy, cx, j, :] = twth
                gt_cls[b, cy, cx, j, int(label[n_obj].item()) - 1] = 1

            pred_xy_ = pred_xy[b]
            pred_wh_ = pred_wh[b]
            center_pred_xy = center_anchors[..., :2] + pred_xy_                     # [845, 2]
            center_pred_wh = center_anchors[..., 2:] * pred_wh_                     # [845, 2]
            center_pred_bbox = torch.cat([center_pred_xy, center_pred_wh], dim=-1)
            corner_pred_bbox = center_to_corner(center_pred_bbox).view(-1, 4)       # [845, 4]

            iou_pred_gt = find_jaccard_overlap(corner_pred_bbox, corner_gt_box_13)  # [845, # obj]
            iou_pred_gt = iou_pred_gt.view(out_size, out_size, 5, -1)
            gt_conf[b] = iou_pred_gt.max(-1)[0]

        return resp_mask, gt_xy, gt_wh, gt_conf, gt_cls

    def forward(self, pred_targets, gt_boxes, gt_labels):
        """

        :param pred_targets: (B, 13, 13, 125)
        :param gt_boxes:     (B, 4)
        :param gt_labels:
        :return:
        """
        out_size = pred_targets.size(1)
        # print("output_size :", out_size)
        pred_targets = pred_targets.view(-1, out_size, out_size, 5, 5 + 20)
        pred_xy = pred_targets[..., :2].sigmoid()                  # sigmoid(tx ty)  0, 1
        pred_wh = pred_targets[..., 2:4].exp()                     # 2, 3 || original yolo loss only exp() 1/2.7 ~ 2.7
        pred_conf = pred_targets[..., 4].sigmoid()                 # 4
        pred_cls = pred_targets[..., 5:]                           # 20

        resp_mask, gt_xy, gt_wh, gt_conf, gt_cls = self.make_target(gt_boxes, gt_labels, pred_xy, pred_wh)

        # 1. xy sse
        xy_loss = resp_mask.unsqueeze(-1).expand_as(gt_xy) * (gt_xy - pred_xy.cpu()) ** 2

        # 2. wh loss
        wh_loss = resp_mask.unsqueeze(-1).expand_as(gt_wh) * (torch.sqrt(gt_wh) - torch.sqrt(pred_wh.cpu())) ** 2

        # 3. conf loss
        conf_loss = resp_mask * (gt_conf - pred_conf.cpu()) ** 2
        # print("gt_conf's max : ", gt_conf.max(), "pred_conf's max : ", pred_conf.max())

        # 4. no conf loss
        no_resp_mask = 1 - resp_mask
        no_conf_loss = no_resp_mask.squeeze(-1) * (gt_conf - pred_conf.cpu()) ** 2

        # 5. classification loss
        pred_cls = F.softmax(pred_cls, dim=-1)  # [N*13*13*5,20]
        # cls_loss = resp_mask.unsqueeze(-1).expand_as(gt_cls) * (gt_cls - pred_cls.cpu()) ** 2  # torch.Size([B, 13, 13, 20])
        cls_loss = resp_mask.unsqueeze(-1).expand_as(gt_cls) * (gt_cls * -1 * torch.log(pred_cls.cpu()))  # soft max loss

        # 6. focal loss
        # ------------------------------- focal loss -----------------------------------
        # gt_cls 에서
        p_t = pred_cls.cpu()
        gamma = 2
        alpha = 0.25

        # balanced cross entropy
        gt_alpha_right_class = torch.full_like(gt_cls, fill_value=alpha)
        gt_alpha_wrong_class = 1 - gt_alpha_right_class
        gt_alpha = torch.where(gt_cls == 1, gt_alpha_right_class, gt_alpha_wrong_class)

        focal_cls_loss = resp_mask.unsqueeze(-1).expand_as(gt_cls) * \
                         (gt_alpha * torch.pow(1-p_t, gamma) * gt_cls * -1 * torch.log(p_t))
        #  나중에 official 이랑 비교해보기
        # ------------------------------- focal loss -----------------------------------

        loss1 = 5 * xy_loss.sum()
        loss2 = 5 * wh_loss.sum()
        loss3 = 1 * conf_loss.sum()
        loss4 = 0.5 * no_conf_loss.sum()
        loss5 = 1 * cls_loss.sum()
        return loss1 + loss2 + loss3 + loss4 + loss5, (loss1, loss2, loss3, loss4, loss5)


if __name__ == '__main__':
    image = torch.randn([5, 3, 416, 416])
    pred = torch.zeros([5, 13, 13, 125])  # batch, 13, 13, etc...

    criterion = Yolo_Loss(num_classes=20)





