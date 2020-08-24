import torch
import torch.nn as nn
from torchvision.models import vgg16_bn
from utils import device


class YOLO_VGG_16(nn.Module):
    def __init__(self, vgg=vgg16_bn(pretrained=True), num_classes=20):
        super().__init__()

        self.num_anchors = 5
        self.num_classes = num_classes

        self.middle_feature = nn.Sequential(*list(vgg.features.children())[:-1])  # after conv5_3 [512, 26, 26]

        self.extra = nn.Sequential(nn.Conv2d(512, 512, 3, 1, 1),
                                   nn.BatchNorm2d(512),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(512, 512, 3, 1, 1),
                                   nn.BatchNorm2d(512),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(512, 512, 3, 1, 1),
                                   nn.BatchNorm2d(512),
                                   nn.ReLU(inplace=True),
                                   nn.MaxPool2d(2, 2))

        self.skip_module = nn.Sequential(nn.Conv2d(512, 64, 1, stride=1, padding=0),
                                         nn.BatchNorm2d(64),
                                         nn.ReLU(inplace=True))

        self.final = nn.Sequential(nn.Conv2d(768, 1024, 3, padding=1),
                                   nn.BatchNorm2d(1024),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(1024, 256, 3, padding=1),
                                   nn.BatchNorm2d(256),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(256, self.num_anchors * (5 + self.num_classes), 1))  # anchor 5, class 20
        self.init_conv2d()
        print("num_params : ", self.count_parameters())

    def init_conv2d(self):
        for c in self.extra.children():
            if isinstance(c, nn.Conv2d):
                nn.init.normal_(c.weight, mean=0.0, std=0.01)
                nn.init.constant_(c.bias, 0.)

        for c in self.skip_module.children():
            if isinstance(c, nn.Conv2d):
                nn.init.normal_(c.weight, mean=0.0, std=0.01)
                nn.init.constant_(c.bias, 0.)

        for c in self.final.children():
            if isinstance(c, nn.Conv2d):
                nn.init.normal_(c.weight, mean=0.0, std=0.01)
                nn.init.constant_(c.bias, 0.)

    def forward(self, x):
        output_size = x.size(-1)
        output_size /= 32
        o_size = int(output_size)

        x = self.middle_feature(x)              # after conv4_3, maxpooling
        skip_x = self.skip_module(x)            # torch.Size([B, 512, 26, 26])--->  torch.Size([B, 64, 26, 26])

        # --------------------- yolo v2 reorg layer ---------------------
        skip_x = skip_x.view(-1, 64, o_size, 2, o_size, 2).contiguous()
        skip_x = skip_x.permute(0, 3, 5, 1, 2, 4).contiguous()
        skip_x = skip_x.view(-1, 256, o_size, o_size)   # torch.Size([B, 256, 13, 13])

        x = self.extra(x)                       # torch.Size([B, 1024, 13, 13])
        x = torch.cat([x, skip_x], dim=1)       # torch.Size([B, 1280, 13, 13])
        x = self.final(x)                       # torch.Size([B, 125, 13, 13])

        return x

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == '__main__':
    model = YOLO_VGG_16().to(device)
    print(model)
    image = torch.randn([1, 3, 416, 416]).to(device)
    print(model(image).size())





