# Step-by-step installation instructions


**a. Create a conda virtual environment named mamba and activate it.**
```bash
conda create -n mamba python=3.10 nvidia/label/cuda-12.1.1::cuda-toolkit
conda activate mamba
```
**b. Clone MamBEV.**
```bash
git clone https://github.com/.../MamBEV.git
```
**c. Run the conda installation script** 
```bash
# note this may take some time
cd ./env_cfg
bash ./install_env.sh
cd ..
```
**OPTIONAL: Install torchviz and graphviz**
```bash
conda install anaconda::graphviz 
pip install torchviz
```

**d. Setup path variable to include local modules**

```bash
pip install -e . 
```

**e. Prepare pretrained models.**
```bash
mkdir ckpts

cd ckpts & wget https://github.com/zhiqi-li/storage/releases/download/v1.0/r101_dcn_fcos3d_pretrain.pth
```

note: this pretrained model is the same model used in [detr3d and bevformer](https://github.com/WangYueFt/detr3d)
