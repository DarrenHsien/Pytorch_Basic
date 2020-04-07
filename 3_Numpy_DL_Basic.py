import numpy as np

# N is batch size; 
# D_in is input dimension;
# H is hidden dimension; 
# D_out is output dimension.
# 64筆data ; 每筆維度:1000
# 輸出10筆結果 ; 隱藏層包含100節點
N, D_in, H, D_out = 64, 1000, 100, 10

# Create random input and output data
x = np.random.randn(N, D_in)
y = np.random.randn(N, D_out)
print("x shape",x.shape)
print("y shape",y.shape)

# Randomly initialize weights
# 1000 -> 100
w1 = np.random.randn(D_in, H)
# 100 -> 10
w2 = np.random.randn(H, D_out)
print("W1 shape",w1.shape)
print("W2 shape",w2.shape)

learning_rate = 1e-6
for t in range(1):
    # 前向傳播Forward pass: compute predicted y
    # x_input -> w1 -> h -> h_relu -> w2 -> y_pred
    # x_input -> w1 -> h
    h = x.dot(w1)
    print("h : ",h.shape)
    # h -> h_relu
    h_relu = np.maximum(h, 0)
    print("h_relu : ",h_relu.shape)
    # h_relu -> w2 -> y_pred
    y_pred = h_relu.dot(w2)
    print("y_pred : ",y_pred.shape)
    

    # Compute and print loss 計算損失ㄐR-square
    loss = np.square(y_pred - y).sum()
    print(t, loss)

    # 反向傳播Backprop to compute gradients of w1 and w2 with respect to loss
    grad_y_pred = 2.0 * (y_pred - y)
    print("grad_y_pred : ",grad_y_pred.shape)
    
    grad_w2 = h_relu.T.dot(grad_y_pred)
    print("grad_w2 : ",grad_w2.shape)
    
    grad_h_relu = grad_y_pred.dot(w2.T)
    
    grad_h = grad_h_relu.copy()
    
    grad_h[h < 0] = 0
    
    grad_w1 = x.T.dot(grad_h)
    print("grad_w1 : ",grad_w1.shape)
    
    # Update weights
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2

"""
Numpy 也可以做DeepLearning
但卻無法與GPU聯動 
"""