# Image-Style-Transfer


Official code : https://github.com/leongatys/PytorchNeuralStyleTransfer

Code to run Neural Style Transfer from paper [Image Style Transfer Using Convolutional Neural Networks](http://www.cv-foundation.org/openaccess/content_cvpr_2016/html/Gatys_Image_Style_Transfer_CVPR_2016_paper.html).

Reference Code

## Setup

docker setup

```console
cd docker
sudo docker build --tag style-transfer .
sudo docker run -it --rm --name "style" 3803fd9102ae python main.py
```

## TODO

- [x] implement vggnet
- [x] import vgg weight from vgg model
- [ ] logger
- [ ] implement Gram Matrix 
- [ ] implement Style Loss
- [ ] implement Contents Loss
- [ ] implement preproessing and postprocessing
- [ ] parameterize for console
- [ ] create notebook for understanding and visualization

- [ ] push docker to dockerhub


## What I learned?

state_dict은 OrderedDict으로 구성되어있어서 위에서 아래로 순서대로 이어져있다. 
물론 forward 쪽에서 어떻게 연결하느냐에 따라 다르겠지만, VGGNET은 매우 단순한 구조로 그냥 선형으로 연결할 수 있기 때문에
그 이름을 변경하는데 있어서 (그냥 쓰면 가독성이 떨어진다고 판단했다.) 아래와 같은 단순한 코드로 이름 변경, 및 load가 가능했다. 
참고로 state_dict은 `{layer_name(str): layer_parameter(Tensor)}` 형으로 구성되어있다.

```python
from collections import OrderedDict
def load_vgg_weight(self):
    import torchvision.models as models
    vgg = models.vgg19(pretrained=True)
    state_dict = []
    for (p_name, p_param), (v_name, v_param) in zip(self.named_parameters(), vgg.named_parameters()):
        state_dict.append((p_name, v_param))
    self.load_state_dict(OrderedDict(state_dict))
```

## enumerate

enumerate를 쓸때는 조심하자. 시작 위치를 정할 수가 없다. 첫번재 것을 빼고 돌리기 위해서 아래와 같은 방법을 사용했다. 

```python 
iter = enumerate(param)
next(iter)
for i, f in iter:
    pass
```

