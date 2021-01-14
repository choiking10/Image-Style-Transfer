import torch
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

        self.conv0 = self.create_layer([3, 64, 64])
        self.conv1 = self.create_layer([64, 128, 128])
        self.conv2 = self.create_layer([128, 256, 256, 256, 256])
        self.conv3 = self.create_layer([256, 512, 512, 512, 512])
        self.conv4 = self.create_layer([512, 512, 512, 512, 512])

    def forward(self, x):
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        return x

    def create_layer(self, param):
        layer = []
        start = param[0]
        iter = enumerate(param)
        next(iter)
        for i, f in iter:
            layer.append((f"c{i-1}", nn.Conv2d(start, f, kernel_size=3, padding=1)))
            layer.append((f"r{i-1}", nn.ReLU(inplace=True)))
            start = f
        layer.append(("p", self.pooling(kernel_size=2, stride=2)))
        return nn.Sequential(OrderedDict(layer))

    def load_vgg_weight(self):
        import torchvision.models as models
        vgg = models.vgg19(pretrained=True)
        state_dict = []
        for (p_name, p_param), (v_name, v_param) in zip(self.named_parameters(), vgg.named_parameters()):
            state_dict.append((p_name, v_param))
        self.load_state_dict(OrderedDict(state_dict))


class GramMatrix(nn.Module):
     def forward(self, x):
         b, c, h, w = x.shape
         F = x.view(-1, c, b * w)
         G = torch.bmm(F, F.transpose(1, 2)) / (h * w)
         return G


class ContentLoss(nn.Module):
    def __init__(self, target_feature):
        super(ContentLoss, self).__init__()
        b, c, h, w = target_feature.shape
        self.target_F = target_feature.view(-1, c, b * w)

    def forward(self, input_feature):
        return nn.MSELoss()(input_feature, self.target_F)


class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.to_gram = GramMatrix()
        self.target_G = self.to_gram(target_feature)

    def forward(self, input_feature):
        return nn.MSELoss()(self.to_gram(input_feature), self.target_G)


def main():
    vgg = VGG("avg")
    vgg.load_vgg_weight()


main()
