# My VRDL HW1

Use ViT_B_16 pretrained model to do classification on bird dataset

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

torch==1.5.1

torchvision==0.6.1

numpy

tqdm

tensorboard

ml-collections


## Training

To train the model(s) in the paper, run this command:

```train
python train.py --input-data <path_to_data> --alpha 10 --beta 20
```

>📋  Describe how to train the models, with example commands on how to train the models in your paper, including the full training procedure and appropriate hyperparameters.

## Evaluation

To evaluate my model on ImageNet, run:

```eval
python eval.py --model-file mymodel.pth --benchmark imagenet
```

>📋  Describe how to evaluate the trained models on benchmarks reported in the paper, give commands that produce the results (section below).

## Pre-trained Models

You can download pretrained models here:

- [My awesome model](https://drive.google.com/mymodel.pth) 

>📋  Give a link to where/how the pretrained models can be downloaded and how they were trained (if applicable).  Alternatively you can have an additional column in your results table with a link to the models.

## Results

Our model achieves the following performance on :


| Model name         | Top 1 Accuracy  | 
| ------------------ |---------------- | 
| My awesome model   |     83%         | 


