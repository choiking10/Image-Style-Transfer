import torch
from torch import nn
from collections import OrderedDict
from PIL import Image
from torch.autograd import Variable

import torchvision
from torchvision import transforms


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


class HookFunc:
    def __init__(self, name):
        self.feature = None
        self.name = name

    def __call__(self, module, inp, out):
        self.feature = out.clone().detach()

    @property
    def data(self):
        return self.feature


class FeatureExtractor:
    def __init__(self, model, feature_names):
        self.data = {}
        named_modules = dict(model.named_modules())
        for name in feature_names:
            self.data[name] = HookFunc(name)
            named_modules[name].register_forward_hook(self.data[name])


def main():
    vgg = VGG("avg")
    vgg.load_vgg_weight()
    style_features = FeatureExtractor(vgg, [f"conv{i}.r0" for i in range(5)])
    content_features = FeatureExtractor(vgg, [f"conv3.r1"])

    vgg.modules()

    img_size = 512
    prep = transforms.Compose([transforms.Scale(img_size),
                               transforms.ToTensor(),
                               transforms.Lambda(lambda x: x[torch.LongTensor([2, 1, 0])]),
                               transforms.Normalize(mean=[0.40760392, 0.45795686, 0.48501961], #subtract imagenet mean
                                                    std=[1, 1, 1]),
                               transforms.Lambda(lambda x: x.mul_(255))
                               ])
    style_image_path = "images/vangogh_starry_night.jpg"
    content_image_path = "images/Tuebingen_Neckarfront.jpg"

    style_img = prep(Image.open(style_image_path))
    content_img = prep(Image.open(content_image_path))

    if torch.cuda.is_available():
        style_img_torch = Variable(style_img.unsqueeze(0).cuda())
        content_img_torch = Variable(content_img.unsqueeze(0).cuda())
    else:
        style_img_torch = Variable(style_img.unsqueeze(0))
        content_img_torch = Variable(content_img.unsqueeze(0))

    opt_img = Variable(content_img_torch.data.clone(), requires_grad=True)

    style_vgg = vgg(style_img_torch)
    content_vgg = vgg(content_img_torch)
    opt_vgg = vgg(opt_img)
    print(vgg)


main()
