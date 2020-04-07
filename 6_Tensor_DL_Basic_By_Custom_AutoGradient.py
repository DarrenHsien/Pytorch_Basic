import torch

# staticmethod : 希望某個函式，完全不要作為實例的綁定方法，也就是不要將第一個參數綁定為所建立的實例，則可以使用@staticmethod加以修飾
class MyReLU(torch.autograd.Function):
    """
    如果我們須自定義我們自己的自動梯度計算函式,我們須繼承torch.autograd.Function 
    同時,我們須在子類別建立
    -forward passes
    -backward passes
    """

    @staticmethod
    def forward(ctx, input):
        """
        h = x.mm(w1)
        等於是把這項導入
        h_relu = h.clamp(min=0)
        y_pred = h_relu.mm(w2)
        """
        ctx.save_for_backward(input)
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        return grad_input


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
    # To apply our Function, we use Function.apply method. We alias this as 'relu'.
    relu = MyReLU.apply

    # 我們可以自定義relu成一個類別並把它加載給autograd,讓torch也可以自己推導relu的前向與反向傳播
    # h = x.mm(w1)
    # relu(h) = h.clamp(min=0)
    y_pred = relu(x.mm(w1)).mm(w2)

    # 計算總體損失張量
    loss = (y_pred - y).pow(2).sum()
    if t % 100 == 99:
        print(t, loss.item())

    # 使用autograde計算反向傳播梯度值.
    loss.backward()

    # 更新權重
    with torch.no_grad():
        w1 -= learning_rate * w1.grad
        w2 -= learning_rate * w2.grad

        # 清理當次梯度值歸0
        w1.grad.zero_()
        w2.grad.zero_()