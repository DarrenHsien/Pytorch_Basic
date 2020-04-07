"""
我們必須手動實現神經網絡的正向和反向傳遞。對於小型的兩層網絡，手動實施反向傳遞並不是什麼大問題，

但是對於大型的複雜網絡而言，可以很快變得非常麻煩。

幸運的是，我們可以使用自動微分 來自動計算神經網絡中的反向通過。

PyTorch中的 autograd軟件包完全提供了此功能。使用autograd時，網絡的前向傳遞將定義一個計算圖；

圖中的節點為張量，邊為從輸入張量生成輸出張量的函數。然後通過該圖進行反向傳播，可以輕鬆計算梯度。

這聽起來很複雜，在實踐中非常簡單。每個張量代表計算圖中的一個節點。假設 x是一個Tensor， x.requires_grad=True然後x.grad是另一個Tensor，它

持有x相對於某個標量值的梯度。

在這裡，我們使用PyTorch張量和autograd來實現我們的兩層網絡。現在我們不再需要手動通過網絡實現反向傳遞：


"""


import torch

dtype = torch.float
# device = torch.device("cpu")
device = torch.device("cuda:0") # Uncomment this to run on GPU

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
# N is batch size; 
# D_in is input dimension;
# H is hidden dimension; 
# D_out is output dimension.
# 64筆data ; 每筆維度:1000
# 輸出10筆結果 ; 隱藏層包含100節點
N, D_in, H, D_out = 64, 1000, 100, 10

# Create random Tensors to hold input and outputs.
# 設定x,y為不可變的張量,可增加cuda在計算上的效能
x = torch.randn(N, D_in, device=device, dtype=dtype)
y = torch.randn(N, D_out, device=device, dtype=dtype)

# Create random Tensors for weights.
# 設定w1,w2為可變的張量
w1 = torch.randn(D_in, H, device=device, dtype=dtype, requires_grad=True)
w2 = torch.randn(H, D_out, device=device, dtype=dtype, requires_grad=True)

learning_rate = 1e-6
for t in range(500):
    """
    h = x.mm(w1)
    h_relu = h.clamp(min=0)
    y_pred = h_relu.mm(w2)
    """
    y_pred = x.mm(w1).clamp(min=0).mm(w2)

    # 計算總體損失 張量
    loss = (y_pred - y).pow(2).sum()
    if t % 100 == 99:
        print(t, loss.item())
    
    # autograd -> 計算反向傳播的數值
    # 會針對requires_grad設定為true的張量進行反向傳播的計算
    # 經過此計算我們可以取得w1與w2應有的grad值
    loss.backward()

    # torch.no_grad() -> 打包了更新權重的方法
    # You can also use torch.optim.SGD to achieve this.
    with torch.no_grad():
        w1 -= learning_rate * w1.grad
        w2 -= learning_rate * w2.grad

        # 須記得將此輪梯度值歸0
        w1.grad.zero_()
        w2.grad.zero_()