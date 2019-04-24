from matplotlib import pyplot as plt
from typing import List
import logging
from typing import Optional
from typing import Tuple
from sklearn.metrics import mean_squared_error
import torch.nn as nn
import numpy as np
import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from torch.autograd import Variable

DATA_FOLDER='./trial'
TRAIN_IMAGES_FOLDER = '../trial/train'
LABELS = '../trial/train_labels.csv'
USE_GPU = torch.cuda.is_available()

logging.basicConfig(level='INFO')
logger = logging.getLogger()

def read_labels(path_to_file: str) -> pd.DataFrame:
    labels = pd.read_csv(path_to_file)
    return labels


def format_labels_for_dataset(labels: pd.DataFrame) -> np.array:
    return labels['label'].values.reshape(-1, 1)


def format_path_to_images_for_dataset(labels: pd.DataFrame, path: str) -> List:
    return [os.path.join(path, f'{f}.tif') for f in labels['id'].values]


class MainDataset(Dataset):
    def __init__(self,
                 x_dataset: Dataset,
                 y_dataset: Dataset,
                 x_tfms: Optional = None):
        self.x_dataset = x_dataset
        self.y_dataset = y_dataset
        self.x_tfms = x_tfms

    def __len__(self) -> int:
        return self.x_dataset.__len__()

    def __getitem__(self, index: int) -> Tuple:
        x = self.x_dataset[index]
        y = self.y_dataset[index]
        if self.x_tfms is not None:
            x = self.x_tfms(x)
        return x, y

class ImageDataset(Dataset):
    def __init__(self, paths_to_imgs: List):
        self.paths_to_imgs = paths_to_imgs

    def __len__(self) -> int:
        return len(self.paths_to_imgs)

    def __getitem__(self, index: int) -> Image.Image:
        img = Image.open(self.paths_to_imgs[index])
        return img

class LabelDataset(Dataset):
    def __init__(self, labels: List):
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> int:
        return self.labels[index]


labels = read_labels(LABELS)
train_labels = format_labels_for_dataset(labels)
train_images = format_path_to_images_for_dataset(labels, TRAIN_IMAGES_FOLDER)
train_images_dataset = ImageDataset(train_images)
train_labels_dataset = LabelDataset(train_labels)

x_tfms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

dataset = MainDataset(train_images_dataset, train_labels_dataset, x_tfms)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])


shuffle = True
batch_size = 25
num_workers = 0
num_epochs = 5
learning_rate = 0.001

train_dataloader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=shuffle,
                              num_workers=num_workers)


test_dataloader = DataLoader(test_dataset,
                              batch_size=batch_size,
                              shuffle=shuffle,
                              num_workers=num_workers)


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
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=5, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer6 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=5, padding=2),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer7 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=5, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer8 = nn.Sequential(
            nn.Conv2d(256, 126, kernel_size=5, padding=2),
            nn.BatchNorm2d(126),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer9 = nn.Sequential(
            nn.Conv2d(126, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer10 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer11 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer12 = nn.Sequential(
            nn.Conv2d(16, 3, kernel_size=5, padding=2),
            nn.BatchNorm2d(3),
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

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)
# -----------------------------------------------------------------------------------
# Train the Model

loss_list = []

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_dataloader ):
        images = Variable(images.float()).cuda()
        labels = Variable(labels.float()).cuda()

        # Forward + Backward + Optimize
        optimizer.zero_grad()
        outputs = cnn(images)
        loss = criterion(outputs, labels)
        loss_list.append(loss.item())
        loss.backward()

        optimizer.step()

        if (i + 1) % 10 == 0:
            print('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f'
                  % (epoch + 1, num_epochs, i + 1, len(train_dataset) // batch_size, loss.item()))



# Test the Model
cnn.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
correct = 0
total = 0
for images, labels in test_dataloader:
    images = Variable(images).cuda()
    outputs = cnn(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted.float().cpu() == labels.float()).sum()
    rms = np.sqrt(mean_squared_error(predicted.float().cpu(), labels.float()))
# -----------------------------------------------------------------------------------

print(rms)

plt.plot(loss_list)
plt.show()

