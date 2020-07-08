import torch
import argparse
import visdom
from dataset.voc_dataset import VOC_Dataset
from torch.utils.data import DataLoader
import torch.optim as optim
from model import YOLO_VGG_16
from loss import Yolo_Loss
import time
import os
from torch.optim.lr_scheduler import StepLR, MultiStepLR
from train import train
from test import test


def main():
    # 1. argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=1e-6)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--save_file_name', type=str, default='yolo_v2_vgg_16')
    parser.add_argument('--conf_thres', type=float, default=0.1)
    parser.add_argument('--save_path', type=str, default='./saves')
    parser.add_argument('--start_epoch', type=int, default=170)  # to resume

    opts = parser.parse_args()
    print(opts)

    # 2. device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 3. visdom
    vis = visdom.Visdom()

    # 4. dataset
    train_set = VOC_Dataset(root="D:\Data\VOC_ROOT\TRAIN", split='TRAIN')
    test_set = VOC_Dataset(root="D:\Data\VOC_ROOT\TEST", split='TEST')

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
    model = YOLO_VGG_16().to(device)

    # 7. criterion
    criterion = Yolo_Loss(num_classes=20)
    # 8. optimizer
    optimizer = optim.SGD(params=model.parameters(),
                          lr=opts.lr,
                          momentum=0.9,
                          weight_decay=5e-4)

    # 9. scheduler
    scheduler = MultiStepLR(optimizer=optimizer, milestones=[120], gamma=0.1)
    scheduler_rate = {"0": 1e-5, "5": 1e-4, "80": 1e-5, "110": 1e-6}

    scheduler = None
    scheduler_rate = None

    # 10. resume
    if opts.start_epoch != 0:

        checkpoint = torch.load(os.path.join(opts.save_path, opts.save_file_name) + '.{}.pth.tar'
                                .format(opts.start_epoch - 1))          # 하나 적은걸 가져와서 train
        model.load_state_dict(checkpoint['model_state_dict'])           # load model state dict
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])    # load optim state dict
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
              save_path=opts.save_path,
              save_file_name=opts.save_file_name,
              scheduler_rate=scheduler_rate)

        if scheduler is not None:
            scheduler.step()

        # 12. test
        test(epoch=epoch,
             device=device,
             vis=vis,
             test_loader=test_loader,
             model=model,
             criterion=criterion,
             save_path=opts.save_path,
             save_file_name=opts.save_file_name,
             conf_thres=opts.conf_thres,
             eval=True)


if __name__ == '__main__':
    main()