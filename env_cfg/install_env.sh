#!/bin/bash
# make an environment called mamba or change the env name at the
# top of the torch231.yml file to the env name of your choice
conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install mmcv==2.1
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
conda env update -f torch231.yml
cd ..
pip install -e .
