
import cv2
from pathlib import Path
import requests

DATA_PATH = Path("data")
PATH = DATA_PATH / "mnist"

PATH.mkdir(parents=True, exist_ok=True)

URL = "http://deeplearning.net/data/mnist/"
FILENAME = "mnist.pkl.gz"

if not (PATH / FILENAME).exists():
        content = requests.get(URL + FILENAME).content
        (PATH / FILENAME).open("wb").write(content)

###############################################################################
# This dataset is in numpy array format, and has been stored using pickle,
# a python-specific format for serializing data.

import pickle
import gzip

with gzip.open((PATH / FILENAME).as_posix(), "rb") as f:
        ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding="latin-1")

###############################################################################
# Each image is 28 x 28, and is being stored as a flattened row of length
# 784 (=28x28). Let's take a look at one; we need to reshape it to 2d
# first.

from matplotlib import pyplot
import numpy as np

print(x_train.shape)
cv2.imshow("img",x_train[0].reshape((28, 28)))
cv2.waitKey(0)
cv2.destroyAllWindows()

#------------------------------------------------------------------------------------------------------

import torch
# 原始數據 28*28攤開
print("x_train.shape : ",x_train.shape)
print("y_train.shape : ",y_train.shape)
print("x_valid.shape : ",x_valid.shape)
print("y_valid.shape : ",y_valid.shape)
x_train, y_train, x_valid, y_valid = map(torch.tensor, (x_train, y_train, x_valid, y_valid))
# 轉張量
print("x_train.shape : ",x_train.shape)
print("y_train.shape : ",y_train.shape)
print("x_valid.shape : ",x_valid.shape)
print("y_valid.shape : ",y_valid.shape)
n, c = x_train.shape
print("y_train min , y_train.max")
print(y_train.min(), y_train.max())




def loss_batch(model, loss_func, xb, yb, opt=None):
    loss = loss_func(model(xb), yb)

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), len(xb)

import numpy as np

def fit(epochs, model, loss_func, opt, train_dl, valid_dl):
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_dl:
            loss_batch(model, loss_func, xb, yb, opt)

        model.eval()
        with torch.no_grad():
            losses, nums = zip(*[loss_batch(model, loss_func, xb, yb) for xb, yb in valid_dl]
            )
        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)

        print(epoch, val_loss)





print("-------------------------------------CNN-------------------------------")
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset


"""
使用Pytorch的預定義 Conv2d類作為我們的捲積層。我們定義具有3個卷積層的CNN。每個卷積後跟一個ReLU。
最後，我們執行平均池化。（請注意，這view是PyTorch的numpy版本 reshape）
"""
lr = 0.1
epochs = 2
bs = 64

class Mnist_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(16, 10, kernel_size=3, stride=2, padding=1)

    def forward(self, xb):
        xb = xb.view(-1, 1, 28, 28)
        xb = F.relu(self.conv1(xb))
        xb = F.relu(self.conv2(xb))
        xb = F.relu(self.conv3(xb))
        xb = F.avg_pool2d(xb, 4)
        return xb.view(-1, xb.size(1))


train_ds = TensorDataset(x_train, y_train)
train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True)

valid_ds = TensorDataset(x_valid, y_valid)
valid_dl = DataLoader(valid_ds, batch_size=bs * 2)

loss_func = F.cross_entropy

model = Mnist_CNN()
opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

fit(epochs, model, loss_func, opt, train_dl, valid_dl)

