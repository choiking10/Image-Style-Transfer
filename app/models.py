from torch import nn
from collections import OrderedDict


class VGG(nn.Module):
    def __init__(self, pool='max'):
        super(VGG, self).__init__()
        self.ativ = nn.ReLU
        if pool == 'max':
            self.pooling = nn.MaxPool2d
        elif pool == "avg":
            self.pooling = nn.AvgPool2d
        else:
            assert True, "pooling layer must be selected in ['max', 'avg']"

        self.conv1 = self.create_layer([3, 64, 64])
        self.conv2 = self.create_layer([64, 128, 128])
        self.conv3 = self.create_layer([128, 256, 256, 256])
        self.conv4 = self.create_layer([256, 512, 512, 512, 512])
        self.conv5 = self.create_layer([512, 512, 512, 512, 512])

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        return x

    def create_layer(self, param):
        layer = []
        start = param[0]
        for i, f in enumerate(param, 1):
            layer.append((f"c{i-1}", nn.Conv2d(start, f, kernel_size=3, padding=1)))
            layer.append((f"r{i-1}", nn.ReLU(inplace=True)))
            start = f
        layer.append(("p", self.pooling(kernel_size=2, stride=2)))
        return nn.Sequential(OrderedDict(layer))


print(VGG("avg"))
