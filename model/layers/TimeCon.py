# time consistency

import torch.nn as nn
import torch



class timecon(nn.Module):
    def __init__(self, in_feature):
        super(timecon, self).__init__()
        self.con1 = nn.Conv3d(in_channels=in_feature, out_channels=in_feature, kernel_size=(1, 5, 5), padding=1, bias=True)
        self.con2 = nn.Conv3d(in_channels=in_feature, out_channels=in_feature, kernel_size=(1, 5, 5), padding=1, bias=True)
        self.con3 = nn.Conv3d(in_channels=in_feature, out_channels=in_feature, kernel_size=(1, 5, 5), padding=1, bias=True)

        self.maxpool = nn.MaxPool2d(3, stride=1)
        self.bn = nn.BatchNorm2d(in_feature)
        self.relu = nn.ReLU()


    def forward(self, x):

        x = self.relu(self.maxpool(self.con1(x)))
        # print(x.shape)
        # x = self.bn(x)
        x = self.relu(self.maxpool(self.con2(x)))
        # x = self.bn(x)
        x = self.relu(self.maxpool(self.con3(x)))

        return x

class timecon_plus(nn.Module):

    def __init__(self, num_classes=2, init_width=96, input_channels=1):
        # self.inplanes = init_width
        super(timecon_plus, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, init_width, kernel_size=(5, 5), stride=2, padding=2, bias=False),
            nn.BatchNorm2d(init_width),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(init_width, init_width*2, kernel_size=(3, 3), stride=1, padding=1, bias=False),
            nn.BatchNorm2d(init_width*2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )


        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        x = self.avgpool(x)

        return x
