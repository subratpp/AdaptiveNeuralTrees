# Adaptive Neural Trees

[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE.md)

This repository contains our PyTorch implementation of Adaptive Neural Trees (ANTs).

The code was written by [Ryutaro Tanno](https://rt416.github.io/) and 
supported by [Kai Arulkumaran](http://kaixhin.com). 

[Paper (ICML'19)](http://proceedings.mlr.press/v97/tanno19a.html) | [Video from London ML meetup](https://www.youtube.com/watch?v=v0bv0HTboOg&t=376s)
<img src='images/cover.png' width="600px"/>


# Prerequisites
- Linux or macOS
- Python 2.7
- Anaconda >= 4.5 
- CPU or NVIDIA GPU + CUDA 8.0 

# Installation

- Clone this repo:
```bash
git clone https://github.com/rtanno21609/AdaptiveNeuralTrees.git
cd AdaptiveNeuralTrees
```
- (Optional) create a new Conda environment and activate it:
```bash
conda create -n ANT python=2.7
source activate ANT
```
- Run the following to install required packages.
``` 
bash ./install.sh
```

## Usage

Training is now config-driven. `tree.py` takes only dataset and experiment name,
and loads all model/training hyperparameters from `training_config.py`.

Single run:

```bash
python tree.py --dataset protein --experiment seed1
```

Outputs for each run are written under:

`./experiments/<dataset>/<experiment>/checkpoints/`

Key files:

- `model.pth`: final model checkpoint
- `tree_structures.json`: learned tree structure
- `records.json`: training trajectory and metrics
- `performance.txt`: final soft vs hard (single-path greedy) report including:
  - soft train/valid/test accuracy
  - hard train/valid/test accuracy
  - total training time

Run 5 seeds and aggregate mean/std automatically:

```bash
./train.sh <dataset>
```

Example:

```bash
./train.sh protein
```

This runs experiments with names `seed1` ... `seed5`, reads each seed's
`performance.txt`, and reports:

- soft train/test mean ± std
- hard train/test mean ± std
- average training time across 5 seeds

It also saves aggregated results to:

`./experiments/<dataset>/results_summary.txt`

## Supported Datasets & Recommended Configurations

The codebase supports both image and tabular datasets. Training details are loaded
automatically from `training_config.py` based on dataset code.

### Dataset Codes

- `mnist`
- `cifar10`
- `letter`
- `connect`
- `census`
- `forest`
- `segment`
- `satimages`
- `pendigits`
- `protein`
- `sensit`

Example runs:

```bash
python tree.py --dataset mnist --experiment seed1
python tree.py --dataset cifar10 --experiment seed1
python tree.py --dataset protein --experiment seed1
```

If you want to change model/training hyperparameters, edit the corresponding
dataset entry in `training_config.py`.

**Jupyter Notebooks**

We have also included two Jupter notebooks `./notebooks/example_mnist.ipynb`
and `./notebooks/example_cifar10.ipynb`, which illustrate how this repository 
can be used to train ANTs on MNIST and CIFAR-10 image recognition datasets. 


**Primitive modules**

Defining an ANT amounts to specifying the forms of primitive modules: routers,
transformers and solvers. The table below provides the list of currently implemented
primitive modules. You can try any combination of three
to construct an ANT. 

| Type | Router | Transformer  | Solver |
| ------------- |:-------------:  | :-----------:|:-----:|
| 1     | 1 x Conv + GAP + Sigmoid | Identity function | Linear classifier  |
| 2     | 1 x Conv + GAP + 1 x FC   | 1 x Conv | MLP with 2 hidden layers  |
| 3     | 2 x Conv + GAP + 1 x FC   | 1 x Conv + 1 x MaxPool | MLP with 1 hidden layer |
| 4     | MLP with 1 hidden layer   | Bottleneck residual block ([He et al., 2015](https://arxiv.org/abs/1512.03385)) | GAP + 2 FC layers + Softmax |
| 5     | GAP + 2 x FC layers ([Veit et al., 2017](https://arxiv.org/abs/1711.11503)) | 2 x Conv + 1 x MaxPool | MLP with 1 hidden layer in AlexNet ([layers-80sec.cfg](https://github.com/BVLC/caffe/blob/master/examples/mnist/lenet.prototxt))    |
| 6     | 1 x Conv +  GAP + 2 x FC | Whole VGG13 architecture (without the linear layer) | GAP + 1 FC layers + Softmax  |

For the detailed definitions of respective modules, please see `utils.py` and 
`models.py`. 

## Citation
If you use this code for your research, please cite our ICML paper:
```
@inproceedings{AdaptiveNeuralTrees19,
  title={Adaptive Neural Trees},
  author={Tanno, Ryutaro and Arulkumaran, Kai and Alexander, Daniel and Criminisi, Antonio and Nori, Aditya},
  booktitle={Proceedings of the 36th International Conference on Machine Learning (ICML)},
  year={2019},
}
```

## Acknowledgements
I would like to thank
[Daniel C. Alexander](http://www0.cs.ucl.ac.uk/staff/d.alexander/) at University College London, UK, 
[Antonio Criminisi](https://scholar.google.co.uk/citations?user=YHmzvmMAAAAJ&hl=en/) at Amazon Research, 
and [Aditya Nori](https://www.microsoft.com/en-us/research/people/adityan/) at Microsoft Research Cambridge
for their valuable contributions to this paper. 





