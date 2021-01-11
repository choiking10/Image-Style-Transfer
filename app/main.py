import torchvision.models as models

vgg = models.vgg19(pretrained=True)
print(vgg)
