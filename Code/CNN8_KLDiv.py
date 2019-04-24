import matplotlib
from typing import List
import logging
from typing import Optional
from functools import partial
from typing import Tuple
from typing import Union
from glob import glob
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_squared_error

import torch.nn as nn
import numpy as np
import os
import pandas as pd
import torch
from torch.optim import Adam
from torchvision.models.resnet import BasicBlock
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from PIL import Image
from matplotlib import pyplot as plt
from torchvision.models.resnet import ResNet
from sklearn.metrics import roc_auc_score
from torch import Tensor
from torchvision import transforms
from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler


labels = pd.read_csv('./Input/train_labels.csv')
train_path = './Input/train/'
num_workers=0

tr, val = train_test_split(labels.label, test_size=0.1)

class CancerDataset(Dataset):
    def __init__(self, datafolder, datatype='train', transform = transforms.Compose([transforms.ToTensor()]), labels_dict={}):
        self.datafolder = datafolder
        self.datatype = datatype
        self.image_files_list = [s for s in os.listdir(datafolder)]
        self.transform = transform
        self.labels_dict = labels_dict
        if self.datatype == 'train':
            self.labels = [labels_dict[i.split('.')[0]] for i in self.image_files_list]
        else:
            self.labels = [0 for _ in range(len(self.image_files_list))]

    def __len__(self):
        return len(self.image_files_list)

    def __getitem__(self, idx):
        img_name = os.path.join(self.datafolder, self.image_files_list[idx])
        image = Image.open(img_name)
        image = self.transform(image)
        img_name_short = self.image_files_list[idx].split('.')[0]

        if self.datatype == 'train':
            label = self.labels_dict[img_name_short]
        else:
            label = 0
        return image, label


data_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

data_transforms_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

img_class_dict = {k:v for k, v in zip(labels.id, labels.label)}



dataset_train = CancerDataset(datafolder='./Input/train/', datatype='train', transform=data_transforms, labels_dict=img_class_dict)

batch_size = 32
train_sampler = SubsetRandomSampler(list(tr.index))
valid_sampler = SubsetRandomSampler(list(val.index))


train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers)
valid_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, sampler=valid_sampler, num_workers=num_workers)

num_epochs = 5
learning_rate = 0.001


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc = nn.Linear(524288, 1)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
# -----------------------------------------------------------------------------------
cnn = CNN()
cnn.cuda()

losslist=[]
criterion = nn.KLDivLoss()
optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)
# -----------------------------------------------------------------------------------

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = Variable(images).cuda()
        labels = Variable(labels).cuda()

        # Forward + Backward + Optimize
        optimizer.zero_grad()
        outputs = cnn(images)
        loss = criterion(outputs.view(-1), labels.float())
        loss.backward()
        optimizer.step()
        losslist.append(loss)

        if (i + 1) % 1 == 0:
            print('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f'
                  % (epoch + 1, num_epochs, i + 1, len(dataset_train) // batch_size, loss.item()))


print(len(dataset_train))

# Test the Model
cnn.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
correct = 0
total = 0
for images, labels in valid_loader:
    images = Variable(images).cuda()
    outputs = cnn(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted.float().cpu() == labels.float()).sum()
    rms = np.sqrt(mean_squared_error(predicted.float().cpu(), labels.float()))
# -----------------------------------------------------------------------------------

print(rms)
plt.plot(losslist)

plt.show()

