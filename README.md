# Image-Style-Transfer



Code to run Neural Style Transfer from our paper [Image Style Transfer Using Convolutional Neural Networks](http://www.cv-foundation.org/openaccess/content_cvpr_2016/html/Gatys_Image_Style_Transfer_CVPR_2016_paper.html).

## Setup

docker setup

```console
cd docker
sudo docker build --tag style-transfer .
sudo docker run -it --rm --name "style" 3803fd9102ae python main.py
```

## TODO

- [ ] logger
- [ ] import vgg weight from vgg model
- [ ] implement Gram Matrix 
- [ ] implement Style Loss
- [ ] implement Contents Loss
- [ ] implement preproessing and postprocessing
- [ ] parameterize for console
- [ ] create notebook for understanding and visualization

- [ ] push docker to dockerhub

