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

## My environment
GUP: GTX1060

CUDA: 10.2
cudnn: 7.6.5

## Training

To train the model(s) in the paper, run this command:

```train
python3 train.py --name {name of this train}
```

## Pre-trained Models

You can download pretrained models here:

- [My awesome model](https://drive.google.com/file/d/1vwzQUQb6aSUr1HOE-oM_r4tiBIUiqbnu/view?usp=sharing) 


## Inference
to reproduce submission file 

```Inference
python3 inference.py
```

## Results

Our model achieves the following performance on :


| Model name         | Top 1 Accuracy  | 
| ------------------ |---------------- | 
| My awesome model   |     83%         | 


