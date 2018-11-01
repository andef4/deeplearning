FROM nvidia/cuda:9.2-base

RUN set -ex \
  && apt update \
  && apt install software-properties-common -y \
  && add-apt-repository ppa:deadsnakes/ppa -y \
  && apt update \
  && apt install -y python3.7 python3.7-dev python3-pip \
  && pip3 install virtualenv \
  && virtualenv -p /usr/bin/python3.7 /venv/ \
  && echo "source /venv/bin/activate" >> ~/.bashrc

RUN set -ex \
  && /venv/bin/pip install fastai \
  && /venv/bin/pip install http://download.pytorch.org/whl/cu92/torch-0.4.1.post2-cp37-cp37m-linux_x86_64.whl \
  && /venv/bin/pip install torchvision
