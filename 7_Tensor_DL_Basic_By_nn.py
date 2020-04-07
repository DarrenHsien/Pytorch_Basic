"""
在PyTorch中，該nn程序包達到了相同的目的。該nn包定義了一組Modules

它們大致等效於神經網絡層。模塊接收輸入張量併計算輸出張量，

可以保持內部狀態，例如包含可學習參數的張量。該nn軟件包還定義了一組有用的損失函數，這些函數通常在訓練神經網絡時使用。

在此示例中，我們使用該nn包來實現我們的兩層網絡：

"""


import torch

# N is batch size; 
# D_in is input dimension;
# H is hidden dimension; 
# D_out is output dimension.
# 64筆data ; 每筆維度:1000
# 輸出10筆結果 ; 隱藏層包含100節點
N, D_in, H, D_out = 64, 1000, 100, 10

# Create random Tensors to hold inputs and outputs
# 設定x,y為不可變的張量,可增加cuda在計算上的效能
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

# 使用 nn package 去定義我們的一連串模型層
# nn.Sequential是一個可以導入其他模組的模組包, 串成一個序列供輸入與產出
# 簡易的節點對節點連接層 : torch.nn.Linear : 會自動創造對應 w 與 bias
# torch.nn.ReLU() : Activation Function
model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out),
)

# 採用Mean Squared Error (MSE)作為loss function.-> sum(y-y_pred)^2
loss_fn = torch.nn.MSELoss(reduction='sum')

learning_rate = 1e-4
for t in range(500):
    
    # 前向傳播
    y_pred = model(x)

    # 計算損失
    loss = loss_fn(y_pred, y)
    if t % 100 == 99:
        print(t, loss.item())

    # 預先歸0反向傳播梯度下降空間.
    model.zero_grad()

    # 計算反向傳播
    loss.backward()

    # 更新 weights
    # model.parameters() : 裝載了多層獨立參數（ 權重,梯度下降值 ） 
    with torch.no_grad():
        for param in model.parameters():
            param -= learning_rate * param.grad
















