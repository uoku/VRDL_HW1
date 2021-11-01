import os
import numpy as np
from torchvision import  models, transforms
import torch
import torch.nn as nn
from PIL import Image
from torch.nn import functional as F
import numpy as np
from models.modeling import VisionTransformer, CONFIGS

with open('./2021VRDL_HW1_datasets/testing_img_order.txt') as f:
    test_images = [x.strip() for x in f.readlines()]  # all the testing images

with open('./2021VRDL_HW1_datasets/classes.txt') as f:
    class_ind = [x.strip() for x in f.readlines()]

i=0
submission = []
for img in test_images:  # image order is important to your result
    print(i)
    i+=1
    # predict
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

    config = CONFIGS["ViT-B_16"]
    model = VisionTransformer(config, 224, zero_head=True, num_classes=200)
    model.load_state_dict(torch.load('output/bird_checkpoint.bin'))
    model.to(device)

    test_image = Image.open('2021VRDL_HW1_datasets/testing_images/'+img)


    model.eval()
    with torch.no_grad():
        validation = torch.stack([image_transforms['validation'](test_image).to(device)])

        pred = model(validation)[0]

        preds = torch.argmax(pred, dim=-1)
        preds = preds.cpu().numpy()[0]
        predicted_class = class_ind[preds]
    # #########
    print(predicted_class)
    submission.append([img, predicted_class])

np.savetxt('answer.txt', submission, fmt='%s')

