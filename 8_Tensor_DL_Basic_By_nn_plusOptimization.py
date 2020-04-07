"""
我們經常使用更複雜的優化器（例如AdaGrad，RMSProp，Adam等）來訓練神經網絡。

PyTorch中的軟件包抽象了優化算法的思想，並提供了常用優化算法的實現


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
# 使用優化器 -> 導入 模型參數 與 學習速率
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for t in range(500):
    # Forward pass: compute predicted y by passing x to the model.
    y_pred = model(x)

    # Compute and print loss.
    loss = loss_fn(y_pred, y)
    if t % 100 == 99:
        print(t, loss.item())

    # 預先歸0反向傳播梯度下降空間
    # 沒優化器前 -> model.zero_grad()
    optimizer.zero_grad()

    # 計算反向傳播
    loss.backward()

    # 更新 weights
    # model.parameters() : 裝載了多層獨立參數（ 權重,梯度下降值 ） 
    """
    with torch.no_grad():
        for param in model.parameters():
            param -= learning_rate * param.grad
    """
    optimizer.step()












