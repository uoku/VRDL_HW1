from os import error
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from random import shuffle


def train_model(model, criterion, optimizer, num_epochs=3):
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)

        for phase in ['train', 'validation']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(image_datasets[phase])
            epoch_acc = running_corrects.double() / len(image_datasets[phase])

            print('{} loss: {:.4f}, acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
    return model


def default_loader(path):
    return Image.open(path).convert('RGB')


class Mydataset(Dataset):
    def __init__(self, img_path, label, transform=None, loader=default_loader):
        imgs = []
        for line in label:
            line = line.strip('\n')
            word = line.split()
            imgs.append((img_path+word[0], int(word[1].split('.')[0])-1))
        self.imgs = imgs
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        p, label = self.imgs[index]
        img = self.loader(p)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs)


image_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ]),
    'validation': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])
}

num_class = 200
img_path = './2021VRDL_HW1_datasets/training_images/'
label_path = './2021VRDL_HW1_datasets/training_labels.txt'

labels = open(label_path, 'r')
all_labels = []
for line in labels:
    all_labels.append(line)
shuffle(all_labels)

train_size = int(len(all_labels) * 0.8)
valid_size = len(all_labels) - train_size

train_path = all_labels[:train_size]
valid_path = all_labels[train_size:]


image_datasets = {
    'train':
    Mydataset(img_path, train_path, transform=image_transforms['train']),
    'validation':
    Mydataset(img_path, valid_path, transform=image_transforms['validation'])
}

dataloaders = {
    'train':
    torch.utils.data.DataLoader(image_datasets['train'],
                                batch_size=64,
                                shuffle=True,
                                num_workers=0),  # for Kaggle
    'validation':
    torch.utils.data.DataLoader(image_datasets['validation'],
                                batch_size=64,
                                shuffle=False,
                                num_workers=0)  # for Kaggle
}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# model
model = models.resnet50(pretrained=True).to(device)

for param in model.parameters():
    param.requires_grad = False

model.fc = nn.Sequential(
    nn.Linear(2048, 128),
    nn.ReLU(inplace=True),
    nn.Linear(128, num_class)).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.fc.parameters())
# model


model_trained = train_model(model, criterion, optimizer, num_epochs=500)

torch.save(model_trained.state_dict(), 'models/weights.h5')
