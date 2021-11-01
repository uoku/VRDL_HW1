from torchvision import models, transforms
import torch
import torch.nn as nn
from PIL import Image
from torch.nn import functional as F
import numpy as np
from models.modeling import VisionTransformer, CONFIGS

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

image_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop((224, 224), scale=(0.05, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ]),
    'validation':
        transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
}

test_image = Image.open('2021VRDL_HW1_datasets/testing_images/0001.jpg')
config = CONFIGS["ViT-B_16"]
model = VisionTransformer(config, 224, zero_head=True, num_classes=200)
model.load_state_dict(torch.load('output/bird_checkpoint.bin'))
model.to(device)


model.eval()
with torch.no_grad():
    validation = torch.stack(
        [image_transforms['validation'](test_image).to(device)])

    pred = model(validation)[0]

    preds = torch.argmax(pred, dim=-1)

    print(preds.cpu().numpy()[0])
