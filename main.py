import torch
import argparse
import visdom
from dataset.voc_dataset import VOC_Dataset
from torch.utils.data import DataLoader
import torch.optim as optim
from model import YOLO_VGG_16
from loss import Yolo_Loss
import os
from torch.optim.lr_scheduler import StepLR
from train import train
from test import test
from config import device


def main():
    # 1. argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--conf_thres', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_workers', type=int, default=4)

    parser.add_argument('--save_file_name', type=str, default='yolo_v2_vgg_16')
    parser.add_argument('--save_path', type=str, default='./saves')
    parser.add_argument('--data_path', type=str, default="D:\Data\VOC_ROOT")

    parser.add_argument('--num_classes', type=int, default=20)
    parser.add_argument('--dataset_type', type=str, default='voc', help='which dataset you want to use VOC or COCO')

    parser.add_argument('--start_epoch', type=int, default=0)  # to resume
    opts = parser.parse_args()
    print(opts)

    # 3. visdom
    vis = visdom.Visdom()

    # 4. dataset
    if opts.dataset_type == 'voc':
        train_set = VOC_Dataset(root=opts.data_path, split='TRAIN')
        test_set = VOC_Dataset(root=opts.data_path, split='TEST')

    # elif opts.dataset_type == 'coco':
    #     train_set = COCO_Dataset(set_name='val2017', split='TRAIN')
    #     test_set = COCO_Dataset(set_name='val2017', split='TEST')

    # 5. dataloader
    train_loader = DataLoader(dataset=train_set,
                              batch_size=opts.batch_size,
                              collate_fn=train_set.collate_fn,
                              shuffle=True,
                              pin_memory=True,
                              num_workers=opts.num_workers)

    test_loader = DataLoader(dataset=test_set,
                             batch_size=1,
                             collate_fn=test_set.collate_fn,
                             shuffle=False)

    # 6. model
    model = YOLO_VGG_16(num_classes=opts.num_classes).to(device)

    # 7. criterion
    criterion = Yolo_Loss(num_classes=opts.num_classes)

    # 8. optimizer
    optimizer = optim.SGD(params=model.parameters(),
                          lr=opts.lr,
                          momentum=0.9,
                          weight_decay=5e-4)

    # 9. scheduler
    scheduler = StepLR(optimizer=optimizer, step_size=100, gamma=0.1)
    # scheduler = None

    # 10. resume
    if opts.start_epoch != 0:

        checkpoint = torch.load(os.path.join(opts.save_path, opts.save_file_name) + '.{}.pth.tar'
                                .format(opts.start_epoch - 1))           # train
        model.load_state_dict(checkpoint['model_state_dict'])            # load model state dict
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])    # load optim state dict
        if scheduler is not None:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])    # load sched state dict
        print('\nLoaded checkpoint from epoch %d.\n' % (int(opts.start_epoch) - 1))

    else:

        print('\nNo check point to resume.. train from scratch.\n')

    # 11. train
    for epoch in range(opts.start_epoch, opts.epochs):

        train(epoch=epoch,
              device=device,
              vis=vis,
              train_loader=train_loader,
              model=model,
              criterion=criterion,
              optimizer=optimizer,
              scheduler=scheduler,
              opts=opts)

        if scheduler is not None:
            scheduler.step()

        test(epoch=epoch,
             device=device,
             vis=vis,
             test_loader=test_loader,
             model=model,
             criterion=criterion,
             opts=opts,
             eval=True)


if __name__ == '__main__':
    main()