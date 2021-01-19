import torch
from torch import nn
from collections import OrderedDict
from PIL import Image
from torch.autograd import Variable

import torchvision
from torchvision import transforms
from torch import optim


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
        self.feature_extractor = {}
        self.feature = []

    def forward(self, x, layers: list):
        #extractor = FeatureExtractor(self, layers)
        #
        # x = self.conv0(x)
        # x = self.conv1(x)
        # x = self.conv2(x)
        # x = self.conv3(x)
        # x = self.conv4(x)

        # extractor.clear_hook()
        ret = [None for i in range(len(layers))]

        for name, module in filter(lambda n:  "." in n[0], self.named_modules()):
            x = module(x)
            if name in layers:
                ret[layers.index(name)] = x
        return list(ret)

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
        self.target_F = target_feature

    def forward(self, input_feature):
        loss = 0
        for target, inp in zip(self.target_F, input_feature):
            loss += nn.MSELoss()(target, inp)
        return loss


class StyleLoss(nn.Module):
    def __init__(self, target_feature, weights=None):
        super(StyleLoss, self).__init__()
        self.gram_mat = GramMatrix()
        self.weights = []
        self.target_G = [self.gram_mat(f) for f in target_feature]

    def forward(self, input_feature):
        input_G = [self.gram_mat(f) for f in input_feature]

        loss = 0
        for target, inp in zip(self.target_G, input_G):
            c, h, w = target.shape
            loss += nn.MSELoss()(target, inp) / (h * w)
        return loss


class HookFunc:
    def __init__(self, name):
        self.feature = None
        self.name = name

    def __call__(self, module, inp, out):
        self.feature = out

    @property
    def data(self):
        return self.feature


class FeatureExtractor:
    def __init__(self, model, feature_names):
        self._data = {}
        self._hook = {}
        named_modules = dict(model.named_modules())
        for name in feature_names:
            self._data[name] = HookFunc(name)
            self._hook[name] = named_modules[name].register_forward_hook(self._data[name])

    def clear_hook(self):
        for name, hook in self._hook.items():
            hook.remove()

    @property
    def data(self):
        return [hook_func.feature for name, hook_func in self._data.items()]


def main():
    vgg = VGG("avg")
    vgg.load_vgg_weight()

    img_size = 512
    prep = transforms.Compose([transforms.Scale(img_size),
                               transforms.ToTensor(),
                               transforms.Lambda(lambda x: x[torch.LongTensor([2, 1, 0])]),
                               transforms.Normalize(mean=[0.40760392, 0.45795686, 0.48501961], #subtract imagenet mean
                                                    std=[1, 1, 1]),
                               transforms.Lambda(lambda x: x.mul(255))
                               ])
    postpa = transforms.Compose([transforms.Lambda(lambda x: x.mul(1. / 255)),
                                 transforms.Normalize(mean=[-0.40760392, -0.45795686, -0.48501961],  # add imagenet mean
                                                      std=[1, 1, 1]),
#                                 transforms.Lambda(lambda x: x[torch.LongTensor([2, 1, 0])]),  # turn to RGB
                                 ])
    postpb = transforms.Compose([transforms.ToPILImage()])

    def postp(tensor):  # to clip results in the range [0,1]
        t = postpa(tensor)
        t[t > 1] = 1
        t[t < 0] = 0
        #img = postpb(t)
        return t

    style_image_path = "images/vangogh_starry_night.jpg"
    content_image_path = "images/Tuebingen_Neckarfront.jpg"

    style_img = prep(Image.open(style_image_path))
    content_img = prep(Image.open(content_image_path))

    if torch.cuda.is_available():
        style_img_torch = Variable(style_img.unsqueeze(0).cuda())
        content_img_torch = Variable(content_img.unsqueeze(0).cuda())
        opt_img = Variable(content_img_torch.data.clone(), requires_grad=True).cuda()
        vgg.cuda()
    else:
        style_img_torch = Variable(style_img.unsqueeze(0))
        content_img_torch = Variable(content_img.unsqueeze(0))
        opt_img = Variable(content_img_torch.data.clone(), requires_grad=True)



    optimizer = optim.Adam([opt_img], lr=10)

    max_iter = 500
    show_iter = 50
    style_alpha = 1e3
    content_beta = 1/1e2
    style_layer = [f"conv{i}.r0" for i in range(5)]
    content_layer = [f"conv3.r1"]

    style_F = vgg(style_img_torch, style_layer)
    content_F = vgg(content_img_torch, content_layer)

    style_F = [f.detach() for f in style_F]
    content_F = [f.detach() for f in content_F]

    style_loss_fn = StyleLoss(style_F)
    content_loss_fn = ContentLoss(content_F)

    for it in range(1, max_iter+1):
        #def closure():
        optimizer.zero_grad()
        features = vgg(opt_img, style_layer + content_layer)
        style_feature, content_feature = features[:len(style_layer)], features[len(style_layer):]
        out_style_loss = style_loss_fn(style_feature)
        out_content_loss = content_loss_fn(content_feature)
        out_content_loss = 0
        loss = style_alpha * out_style_loss + content_beta * out_content_loss
        loss.backward()
        if it % show_iter == 0:
            print(f'Iteration: {it}, loss: {loss.item()}')
        #    return loss
        optimizer.step()

    print("end")

main()
