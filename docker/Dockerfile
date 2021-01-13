FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-devel

MAINTAINER choiking10@gmail.com

RUN apt-get -y update
RUN apt-get -y upgrade
RUN apt install -y python-pip python-dev build-essential vim wget

RUN wget https://download.pytorch.org/models/vgg19-dcbb9e9d.pth -P /root/.cache/torch/hub/checkpoints/
# COPY requirements.txt /tmp/
# RUN pip install -r /tmp/requirements.txt

#COPY app/build /tmp/build/
#WORKDIR /tmp/build/
#RUN python setup.py install


## APP
#COPY app /app
#WORKDIR /app

# RUN mkdir /data
