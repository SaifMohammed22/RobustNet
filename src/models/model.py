import torch.nn as nn


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=1, stride=1, padding=0, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(out_channels, out_channels *
                               self.expansion, kernel_size=1, stride=1, padding=0, bias=False)
        self.batch_norm3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.i_downsample = i_downsample
        self.stride = stride
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x.clone()
        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.relu(self.batch_norm2(self.conv2(x)))

        x = self.conv3(x)
        x = self.batch_norm3(x)
        if self.i_downsample is not None:
            identity = self.i_downsample(identity)
        x += identity
        x = self.relu(x)
        return x


class Block(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
        super(Block, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels * self.expansion,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.batch_norm2 = nn.BatchNorm2d(out_channels * self.expansion)

        self.relu = nn.ReLU()
        self.i_downsample = i_downsample
        self.stride = stride

    def forward(self, x):
        identity = x.clone()
        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.batch_norm2(self.conv2(x))

        if self.i_downsample is not None:
            identity = self.i_downsample(identity)
        x += identity
        x = self.relu(x)
        return x


class ResNet(nn.Module):
    def __init__(self, ResBlock, layer_list, num_classes, num_channels=3,
                 base_channels=64, use_maxpool=True):
        super(ResNet, self).__init__()
        self.in_channels = base_channels

        self.conv1 = nn.Conv2d(
            num_channels, base_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(base_channels)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) if use_maxpool else nn.Identity()

        channel_list = [base_channels * (2 ** i) for i in range(len(layer_list))]

        self.layers = nn.ModuleList()
        for i, (blocks, planes) in enumerate(zip(layer_list, channel_list)):
            stride = 1 if i == 0 else 2
            self.layers.append(self._make_layer(ResBlock, blocks, planes, stride))

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(channel_list[-1] * ResBlock.expansion, num_classes)

    def forward(self, x):
        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.max_pool(x)
        for layer in self.layers:
            x = layer(x)
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x

    def _make_layer(self, ResBlock, blocks, planes, stride=1):
        ii_downsample = None
        layers = []

        if stride != 1 or self.in_channels != planes * ResBlock.expansion:
            ii_downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, planes *
                          ResBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * ResBlock.expansion),
            )

        layers.append(ResBlock(self.in_channels, planes,
                               i_downsample=ii_downsample, stride=stride))
        self.in_channels = planes * ResBlock.expansion

        for i in range(blocks - 1):
            layers.append(ResBlock(self.in_channels, planes))
        return nn.Sequential(*layers)


def ResNet20(num_classes, channels=3):
    return ResNet(Block, [3, 3, 3], num_classes, channels, base_channels=16, use_maxpool=False)


def ResNet32(num_classes, channels=3):
    return ResNet(Block, [5, 5, 5], num_classes, channels, base_channels=16, use_maxpool=False)


def ResNet44(num_classes, channels=3):
    return ResNet(Block, [7, 7, 7], num_classes, channels, base_channels=16, use_maxpool=False)


def ResNet56(num_classes, channels=3):
    return ResNet(Block, [9, 9, 9], num_classes, channels, base_channels=16, use_maxpool=False)


if __name__ == "__main__":
    import torch
    from torchsummary import summary
    from thop import profile

    for name, model_fn in [("ResNet20", ResNet20), ("ResNet32", ResNet32),
                           ("ResNet44", ResNet44), ("ResNet56", ResNet56)]:
        model = model_fn(100, 3)
        flops, params = profile(model, inputs=(torch.randn(1, 3, 32, 32),), verbose=False)
        print(f"{name:>8} | params: {params/1e6:.2f}M | flops: {flops/1e6:.1f}M")
