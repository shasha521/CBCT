# Removing ring artifacts in CBCT images via Transformer

## Installation Guide

### Prerequisites

Before installing the program, you need to complete the following steps:


- Download the source code from https://github.com/guochengqian/PointNeXt.

- Install it as its guide.


## Package dependencies

The project is built with PyTorch 1.9.0, Python3.7, CUDA11.1. For package dependencies, you can install them by:

```bash
pip install -r requirements.txt
```

## Data preparation 

### Dataset
For training data of real cbct images, you can download the dataset from the [official url](https://tianchi.aliyun.com/competition/entrance/532087/information).

Then put all the data into `../datasets/denoising`.


## Training

To train the model, you can begin the training by:

```sh
sh script/train.sh
```



## Evaluation
To evaluate our model, you can run:

```sh
sh script/test.sh
```
For evaluate on each dataset, you should uncomment corresponding line.


## Acknowledgement

This code borrows heavily from [Uformer](https://vinthony.github.io).


## Contact
Please contact us if there is any question or suggestion(980973357@qq.com).

