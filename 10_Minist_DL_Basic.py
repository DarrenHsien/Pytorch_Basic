"""
PyTorch提供了設計精美的模塊和類torch.nn， torch.optim， Dataset和DataLoader 來幫助您創建和訓練神經網絡。
為了充分利用它們的功能並針對您的問題對其進行自定義，您需要真正了解它們在做什麼。
MNIST數據集上訓練基本神經網絡，而無需使用這些模型的任何功能；我們最初只會使用最基本的PyTorch張量功能。
然後，我們將逐步從增加一個功能torch.nn，torch.optim，Dataset，或 DataLoader在同一時間，
正好顯示每一塊做什麼，以及它如何使代碼或者更簡潔，更靈活。


我們將使用經典的MNIST數據集，該數據集由手繪數字的黑白圖像組成（0到9之間）。

我們將使用pathlib 處理路徑（Python 3標準庫的一部分），並使用request下載數據集 。

我們僅在使用模塊時才導入它們，因此您可以確切地看到每個點正在使用的模塊。
"""
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


#---------------------建立無套件幫助的神經網絡單次batch前項傳播------------------------
import math
#784 - 10 節點間權重
weights = torch.randn(784, 10) / math.sqrt(784)
#可異動（給autograd計算梯度）
weights.requires_grad_()
#bias
# wx + b = y
# 因output為10,因此bias設定10個
bias = torch.zeros(10, requires_grad=True)

def log_softmax(x):
    return x - x.exp().sum(-1).log().unsqueeze(-1)

# @ : 點積運算
# input : xb
# W . xb + bias ->pass through log_softmax(activate) -> output
def model(xb):
    return log_softmax(xb @ weights + bias)

bs = 64  # batch size
# 一次拋進64筆
xb = x_train[0:bs]  # a mini-batch from x
preds = model(xb)
# 輸出的preds[0]極為其中一筆數據 會帶有0～9預測機率,同時也攜帶梯度函數,為了用來反向傳播所準備
print("預測結果 : ",preds[0])
print("目標結果 : ",y_train[0])
print("取出預測結果[目標結果當索引]＊-1當成loss")
#我們開始定義損失函數
def nll(input, target):
    #print(range(target.shape[0]))
    #print(target)
    #print(input[range(target.shape[0]), target])
    # pred每一筆會預測0-9每一個產出的機率
    # 此loss則是把 實際結果5 pred[0][5]拿出並加上負號當成loss
    return -input[range(target.shape[0]), target].mean()

loss_func = nll
yb = y_train[0:bs]
print("此次batch造成平均損失 : ",loss_func(preds, yb))

def accuracy(out, yb):
    preds = torch.argmax(out, dim=1)
    return (preds == yb).float().mean()
print("此次計算之準確度 : ",accuracy(preds, yb))

print("------------------------------------完整訓練兩部+反向傳播--------------------------------------")

lr = 0.5  # learning rate
epochs = 2  # how many epochs to train for

for epoch in range(epochs):
    for i in range((n - 1) // bs + 1):
        #         set_trace()
        start_i = i * bs
        end_i = start_i + bs
        xb = x_train[start_i:end_i]
        yb = y_train[start_i:end_i]
        pred = model(xb)
        loss = loss_func(pred, yb)
        #print(loss)
        loss.backward()
        with torch.no_grad():
            weights -= weights.grad * lr
            bias -= bias.grad * lr
            weights.grad.zero_()
            bias.grad.zero_()
#查看損失與精準度
print(loss_func(model(x_train), y_train), accuracy(model(x_train), y_train))

print("------------------------------------完整訓練兩部+反向傳播+修正loss+剔除activation----------------")
weights.grad.zero_()
bias.grad.zero_()
import math
#784 - 10 節點間權重
weights = torch.randn(784, 10) / math.sqrt(784)
#可異動（給autograd計算梯度）
weights.requires_grad_()
#bias
# wx + b = y
# 因output為10,因此bias設定10個
bias = torch.zeros(10, requires_grad=True)

import torch.nn.functional as F

loss_func = F.cross_entropy

def model(xb):
    return xb @ weights + bias

lr = 0.5  # learning rate
epochs = 2  # how many epochs to train for

for epoch in range(epochs):
    for i in range((n - 1) // bs + 1):
        #         set_trace()
        start_i = i * bs
        end_i = start_i + bs
        xb = x_train[start_i:end_i]
        yb = y_train[start_i:end_i]
        pred = model(xb)
        loss = loss_func(pred, yb)
        #print(loss)
        loss.backward()
        with torch.no_grad():
            weights -= weights.grad * lr
            bias -= bias.grad * lr
            weights.grad.zero_()
            bias.grad.zero_()
#查看損失與精準度
print(loss_func(model(x_train), y_train), accuracy(model(x_train), y_train))

print("------------------------------------------------------自定義模型-------------------------------")
from torch import nn

class Mnist_Logistic(nn.Module):
    
    # 初始化會在farward以及back會用到的資料
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(784, 10) / math.sqrt(784))
        self.bias = nn.Parameter(torch.zeros(10))

    def forward(self, xb):
        return xb @ self.weights + self.bias

model = Mnist_Logistic()
loss_func = F.cross_entropy

lr = 0.5  # learning rate
epochs = 2  # how many epochs to train for


def fit():
    for epoch in range(epochs):
        for i in range((n - 1) // bs + 1):
            start_i = i * bs
            end_i = start_i + bs
            xb = x_train[start_i:end_i]
            yb = y_train[start_i:end_i]
            pred = model(xb)
            loss = loss_func(pred, yb)

            loss.backward()
            with torch.no_grad():
                for p in model.parameters():
                    p -= p.grad * lr
                model.zero_grad()

fit()
model.eval()
print(loss_func(model(x_train), y_train), accuracy(model(x_train), y_train))


print("------------------------------------自定義模型 ： 使用nn重構--------------------------------------")
class Mnist_Logistic(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(784, 10)

    def forward(self, xb):
        return self.lin(xb)

model = Mnist_Logistic()
loss_func = F.cross_entropy

lr = 0.5  # learning rate
epochs = 2  # how many epochs to train for
fit()
model.eval()
print(loss_func(model(x_train), y_train), accuracy(model(x_train), y_train))


print("------------------------------------自定義模型 ： 使用nn重構+優化器-----------------------------------")
from torch import optim

def get_model():
    model = Mnist_Logistic()
    return model, optim.SGD(model.parameters(), lr=lr)

model, opt = get_model()

lr = 0.5  # learning rate
epochs = 2  # how many epochs to train for

for epoch in range(epochs):
    for i in range((n - 1) // bs + 1):
        start_i = i * bs
        end_i = start_i + bs
        xb = x_train[start_i:end_i]
        yb = y_train[start_i:end_i]
        pred = model(xb)
        loss = loss_func(pred, yb)

        loss.backward()
        opt.step()
        opt.zero_grad()
model.eval()
print(loss_func(model(x_train), y_train), accuracy(model(x_train), y_train))


print("------------------------------------自定義模型 ： 使用nn重構+優化器+搭配具切片功能資料集模組-----------------------------------")
"""
PyTorch的TensorDataset 是一個包裝張量的數據集。
通過定義索引的長度和方式，這也為我們提供了沿張量的第一維進行迭代，索引和切片的方法。
這將使我們在訓練的同一行中更容易訪問自變量和因變量。
"""
from torch.utils.data import TensorDataset

train_ds = TensorDataset(x_train, y_train)

model, opt = get_model()

for epoch in range(epochs):
    for i in range((n - 1) // bs + 1):
        xb, yb = train_ds[i * bs: i * bs + bs]
        pred = model(xb)
        loss = loss_func(pred, yb)

        loss.backward()
        opt.step()
        opt.zero_grad()

model.eval()
print(loss_func(model(x_train), y_train), accuracy(model(x_train), y_train))





print("------------------------------------自定義模型 ： 使用nn重構+優化器+搭配DataLoader-----------------------------------")

from torch.utils.data import DataLoader

train_ds = TensorDataset(x_train, y_train)
train_dl = DataLoader(train_ds, batch_size=bs)

model, opt = get_model()

for epoch in range(epochs):
    for xb, yb in train_dl:
        pred = model(xb)
        loss = loss_func(pred, yb)

        loss.backward()
        opt.step()
        opt.zero_grad()

model.eval()
print(loss_func(model(x_train), y_train), accuracy(model(x_train), y_train))


print("------------------------------------自定義模型：使用nn重構+優化器+搭配DataLoader+印證-----------------------------------")



train_ds = TensorDataset(x_train, y_train)
train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True)

valid_ds = TensorDataset(x_valid, y_valid)
valid_dl = DataLoader(valid_ds, batch_size=bs * 2)

model, opt = get_model()

for epoch in range(epochs):
    # 調校模行為訓練模式
    model.train()
    for xb, yb in train_dl:
        pred = model(xb)
        loss = loss_func(pred, yb)

        loss.backward()
        opt.step()
        opt.zero_grad()

    # 調校模行為預測模式
    model.eval()
    with torch.no_grad():
        valid_loss = sum(loss_func(model(xb), yb) for xb, yb in valid_dl)

    print(epoch, valid_loss / len(valid_dl))
print(loss_func(model(x_train), y_train), accuracy(model(x_train), y_train))


print("------------------------------------自定義模型：使用nn重構+優化器+搭配DataLoader+印證-----------------------------------")

"""
由於我們經歷了兩次相似的過程來計算訓練集和驗證集的損失，因此我們將其設為自己的函數loss_batch，該函數計算一批的損失
"""
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

def get_data(train_ds, valid_ds, bs):
    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True),
        DataLoader(valid_ds, batch_size=bs * 2),
    )

train_dl, valid_dl = get_data(train_ds, valid_ds, bs)
model, opt = get_model()
fit(epochs, model, loss_func, opt, train_dl, valid_dl)

print(loss_func(model(x_train), y_train), accuracy(model(x_train), y_train))





