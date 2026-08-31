<div align="center">

# Robust~~Ness~~Net

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>

</div>

## Description
This is a CNN model with Conformal Prediction (CP) for robust classification task with prediction set instead of point estimate.

### What am I going to do?
In this project I will:
* Try to build a CNN (ResNet) model from scratch and train/test it using the best practices on CIFAR dataset, or use a pre-trained model and fine-tune it with a new dataset
* Add a Conformal Prediction layer to get prediction set that gives a guarantee that the true label is within the prediction set with a certain, predefined percentage (e.g., 90%)

You can read more about CP from this amazing [paper](https://arxiv.org/pdf/2107.07511). 


## How to run

Install dependencies

```bash
# clone project
git clone https://github.com/SaifMohammed22/RobustNet.git
cd RobustNet

# [OPTIONAL] create conda environment
conda create -n myenv
conda activate myenv

# install pytorch according to instructions
# https://pytorch.org/get-started/

# install requirements
pip install -r requirements.txt
```

Train model with default configuration

```bash
python3 src/train.py
```