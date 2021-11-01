import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from PIL import Image
from matplotlib import pyplot as plt
import torchvision.models as models
from torchsummary import summary
from torch.utils.data import DataLoader, Dataset, random_split


def train_and_valid(model, loss_func, optimizer, epochs=25):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    history = []
    best_acc = 0.0
    best_epoch = 0

    for epoch in range(epochs):
        epoch_start = time.time()
        print("Epoch: {}/{}".format(epoch+1, epochs))

        model.train()

        train_loss = 0.0
        train_acc = 0.0
        valid_loss = 0.0
        valid_acc = 0.0

        for i, (inputs, labels) in enumerate(train_data):
            inputs = inputs.to(device)

            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)

            loss = loss_func(outputs, labels)

            loss.backward()

            optimizer.step()
            train_loss += loss.item() * inputs.size(0)

        with torch.no_grad():
            model.eval()

            for j, (inputs, labels) in enumerate(valid_data):
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)

                loss = loss_func(outputs, labels)

                valid_loss += loss.item() * inputs.size(0)

                ret, predictions = torch.max(outputs.data, 1)
                correct_count = predictions.eq(
                    labels.data.view_as(predictions))

                acc = torch.mean(correct_count.type(torch.FloatTensor))

                valid_acc += acc.item() * inputs.size(0)

        avg_train_loss = train_loss/train_size
        avg_train_acc = train_acc/train_size

        avg_valid_loss = valid_loss/valid_size
        avg_valid_acc = valid_acc/valid_size

        history.append([avg_train_loss, avg_valid_loss,
                       avg_train_acc, avg_valid_acc])

        if best_acc < avg_valid_acc:
            best_acc = avg_valid_acc
            best_epoch = epoch + 1

        epoch_end = time.time()

        print("Epoch: {:03d}, Training: Loss: {:.4f}, Accuracy: {:.4f}%, \n\t\tValidation: Loss: {:.4f}, Accuracy: {:.4f}%, Time: {:.4f}s".format(
            epoch+1, avg_train_loss, avg_train_acc *
            100, avg_valid_loss, avg_valid_acc*100, epoch_end-epoch_start
        ))
        print("Best Accuracy for validation : {:.4f} at epoch {:03d}".format(
            best_acc, best_epoch))

        torch.save(model, './models/model_'+str(epoch+1)+',pt')

    return model, history


def default_loader(path):
    return Image.open(path).convert('RGB')


class Mydataset(Dataset):
    def __init__(self, img_path, label_path, transform=None, loader=default_loader):
        label = open(label_path, 'r')
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
        transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
        transforms.RandomRotation(degrees=15),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
}

img_path = './2021VRDL_HW1_datasets/training_images/'
label_path = './2021VRDL_HW1_datasets/training_labels.txt'

train_data = Mydataset(img_path, label_path,
                       transform=image_transforms['train'])

train_size = int(len(train_data) * 0.8)
valid_size = len(train_data) - train_size

train_data, valid_data = random_split(train_data, [train_size, valid_size])

train_data = DataLoader(train_data, batch_size=10, shuffle=True)
valid_data = DataLoader(valid_data, batch_size=10, shuffle=True)


num_class = 200


res101 = models.resnet101(pretrained=True)
# for param in res101.parameters():
#    param.requires_grad = False

num_fit = res101.fc.in_features
res101.fc = nn.Sequential(
    nn.Linear(num_fit, 256),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(256, num_class),
    nn.LogSoftmax(dim=1)
)

res101 = res101.to('cuda:0')

loss_func = nn.NLLLoss()
optimizer = optim.Adam(res101.parameters())

num_epochs = 10

train_model, history = train_and_valid(
    res101, loss_func, optimizer, num_epochs)
torch.save(history, './models/_history.pt')
