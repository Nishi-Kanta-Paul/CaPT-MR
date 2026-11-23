import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        return self.layers(x)


class Conv64(nn.Module):
    channels = 64

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(ConvBlock(3, 64), ConvBlock(64, 64), ConvBlock(64, 64), ConvBlock(64, 64))

    def forward(self, x):
        return self.layers(x)


class ResidualStage(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        layers = []
        for i in range(3):
            layers += [nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, 3, padding=1, bias=False),
                       nn.BatchNorm2d(out_channels)]
            if i < 2:
                layers.append(nn.ReLU(inplace=True))
        self.body = nn.Sequential(*layers)
        self.shortcut = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels))
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        return self.pool(self.relu(self.body(x) + self.shortcut(x)))


class ResNet12(nn.Module):
    channels = 640

    def __init__(self, widths=(64, 160, 320, 640)):
        super().__init__()
        stages, previous = [], 3
        for width in widths:
            stages.append(ResidualStage(previous, width))
            previous = width
        self.layers = nn.Sequential(*stages)

    def forward(self, x):
        return self.layers(x)


BACKBONES = {'conv64': Conv64, 'resnet12': ResNet12}


def build_backbone(name):
    return BACKBONES[name]()
