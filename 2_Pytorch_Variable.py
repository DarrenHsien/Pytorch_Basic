"""
Variable是PyTorch這個套件的核心物件，我們更新模型的方式完全體現在它的設計中。
它不僅用來儲存資料和模型中的變數，也可以直接用它來計算與儲存梯度（gradient）。

一個Variable最重要的屬性（attribute）是data，它是一個Tensor物件，儲存這個變數現在的值。
"""


#--------------------------Pytorch Variable----------------------------

import torch
from torch.autograd import Variable
"""
所有的Tensor都可以加給於Variable上
注意！！
tensor的操作是單純的資料修改，沒有紀錄
Variable的操作除了data的資料會有改動，所有的操作也會記錄下來變成一個有向圖，藉由creator這個屬性儲存起來。
Variable還有兩個重要的屬性。

-requires_grad
指定要不要更新這個變數，對於不需要更新的變數可以把他設定成False，可以加快運算。
-volatile
指定需不需要保留紀錄用的變數。指定變數為True代表運算不需要記錄，可以加快運算。如果一個變數的volatile是True，則它的requires_grad一定是False。

簡單來說，對於需要更新的Variable記得將requires_grad設成True，當只需要得到結果而不需要更新的Variable可以將volatile設成True加快運算速度
"""
m1 = torch.ones(5, 3)
m2 = torch.ones(5, 3)
a = Variable(m1)
b = Variable(m2)
c = a + b
print(c.data)

#--------------------------AutoGrad----------------------------
"""
如同上面所說的，反向傳播是我們現在廣泛使用的更新模型方式。
當我們定義了誤差如何計算的同時，其實也隱含定義了反向傳播的傳遞方向。
這正是Autograd的運作原理：藉由前面所說的有向圖，PyTorch可以自動幫我們計算梯度。
我們只要對於誤差的Variable物件呼叫backward函數，就可以把沿途所用到參數的gradient都計算出來，儲存在各個參數的grad屬性裡。
最後，更新每個參數的data值。通常，我們使用優化器（optimizer）來更新它們。

常用的優化器有非常多種：SGD、RMSprop、Adagrad、Adam等等，其中最基本的是SGD，代表Stochastic Gradient Descent
"""
import torch
from torch.autograd import Variable
from torch.optim import SGD

m1 = torch.ones(5, 3)
m2 = torch.ones(5, 3)

# 記得要將requires_grad設成True
a = Variable(m1, requires_grad=True)
b = Variable(m2, requires_grad=True)

# 初始化優化器，使用SGD這個更新方式來更新a和b
optimizer = SGD([a, b], lr=0.1)

"""
操作現有的參數與輸入的變數，得到預測。利用預測和正確答案定義我們的誤差。
呼叫優化器的zero_grad將上次更新的梯度歸零。
呼叫誤差的backward算出所有參數的梯度。
呼叫優化器的step更新參數
"""
for _ in range(10):        # 我們示範更新10次
    loss = (a + b).sum()   # 假設a + b就是我們的loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()       # 更新
    #print(loss)


#--------------------------Layers----------------------------
"""
深度學習的模型常常用一層一層的layer來作為變數操作的單位。
Layer又是五花八門，常用的有Full-connected layer，Convolutional layer、Recurrent layer等等。
每一種layer通常包含不只一個Variable的操作。

#只需要定義"模組創建的時候用到的參數"，以及"模組從輸入到輸出做了怎樣的操作"

#input -> conv2d -> relu -> conv2d -> relu

"""
import torch.nn as nn
import torch.nn.functional as F
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, 5)    # 註冊了conv1這個名字
        self.conv2 = nn.Conv2d(20, 20, 5)   # 註冊了conv2這個名字

    def forward(self, x):
       x = F.relu(self.conv1(x))
       return F.relu(self.conv2(x))
       
print(Model())    # 會印出4個參數'conv1.weight', 'conv1.bias',
                               # 'conv2.weight', 'conv2.bias'的值

# 假設 _input是一個變數
model = Model()
# y就是我們模組的輸出
#y = model(_input)    




#--------------------------cpu/CUDA----------------------------
"""
將資料搬到CPU/GPU之前提過PyTorch支援GPU運算。
Module可以讓我們一次把所有包含的變數一次搬到CPU/GPU。
注意到兩個Tensor的運算只能在同一個CPU/GPU上執行，所以將所有變數一次搬移是個很重要的功能。
呼叫cpu()和cuda()可以執行這個功能。
另外，我們可以用torch.cuda.is_available()來檢查我們可不可以使用CUDA來運算。
"""
model = Model()
if torch.cuda.is_available():
    model.cuda()


#--------------------------訓練/運算模式----------------------------
"""
有很多模組在訓練的時候和預測的時候用到同樣的參數，但是執行的運算不一樣，例如Dropout、Batch Normalization等。
因此在訓練和運算的時候，記得分別呼叫train()和eval()來切換模式。

一般來說，我們會分別用不同的函式來包裝訓練和預測的功能。所以一個典型的程式會長的像下面這個樣子。
"""
def update(model, loader):
    model.train()
    # ...

def evaluate(model, loader):
    model.eval()


#--------------------------儲存/載入模型----------------------------
"""
當我們訓練完一個模型，最重要的當然是把它儲存起來在日後使用。
當我們呼叫state_dict()，會拿到一個參數名稱對應到值的字典，然後我們可以呼叫PyTorch的內建函式把它儲存起來。
"""
torch.save(model.state_dict(), PATH)

"""
日後要拿回來的時候，可以呼叫load_state_dict把值載入到對應的參數名稱。
"""
model.load_state_dict(torch.load(PATH))










