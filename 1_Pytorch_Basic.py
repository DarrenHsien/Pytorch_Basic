import torch
import numpy as np
#創造矩陣
m = torch.ones(3, 3)    # 創造一個填滿1的矩陣
print("創造填滿1的矩陣 m : \n",m)

m = torch.zeros(3, 3)   # 創造一個填滿0的矩陣
print("創造填滿0的矩陣 m : \n",m)

m = torch.eye(3)        # 創造一個4x4的單位矩陣
print("創造4x4矩陣單位矩陣 m : \n",m)

m = torch.rand(3, 3)    # 創造一個元素在[0,1)中隨機分佈的矩陣
print("創造3x3(0~1)矩陣 m : \n",m)
m = torch.randn(3, 3)   # 創造一個元素從常態分佈(0, 1)隨機取值的矩陣
print("創造3x3(0~1)矩陣 m : \n",m)


#矩陣操作
m1 = torch.zeros(4, 3)   # 創造一個填滿0的矩陣
m2 = torch.ones(4, 3)   # 創造一個填滿0的矩陣
m = torch.cat((m1, m2), 1)    # 將m1和m2兩個矩陣在第一個維度合併起來
print("將m1和m2兩個矩陣在第一個維度合併起來 : \n",m)
print("m1 shape  : \n",m1.shape)
print("m2 shape  : \n",m2.shape)
print("m shape  : \n",m.shape)
m = torch.stack((m1, m2), 1)  # 將m1和m2兩個矩陣在新的維度（第一維）疊起來
print("將m1和m2兩個矩陣在新的維度（第一維）疊起來 : \n",m)
print("m1 shape  : \n",m1.shape)
print("m2 shape  : \n",m2.shape)
print("m shape  : \n",m.shape)


# m的第一維多一個維度，即
# (A, B) -> (A, 1, B)
print("m1的第一維多一個維度 : \n",m1.unsqueeze(1))
print("m1.unsqueeze(1) shape  : \n",m1.unsqueeze(1).shape)
m1 = m1.unsqueeze(1)

# 如果m的第一維的長度是1，則合併這個維度，即
# (A, 1, B) -> (A, B)
print("如果m1的第一維的長度是1，則合併這個維度 : \n",m1.squeeze(1))
print("m1.squeeze(1) shape  : \n",m1.squeeze(1).shape)
m1 = m1.squeeze(1)


 
# 矩陣element-wise相加，其他基本運算是一樣的
print("m1 shape  : \n",m1.shape)
print("m2 shape  : \n",m2.shape)
m = m1 + m2                   
print("m shape  : \n",m.shape)
print("矩陣element-wise相加 : \n",m)
#其他重要操作

# view : 重構張量維度
print("m.view : \n",m.view(4,3,-1))
print("m.view shape : \n",m.view(4, 3, -1).shape)

# 將m擴展到(1,4, 3)的大小
print("m.expand : \n",m.expand(1,4, 3))
print("m.expand shape : \n",m.expand(1,4,3).shape)

m.cuda()            # 將m搬移到GPU來運算
m.cpu()             # 將m搬移到CPU來運算


n = np.zeros((3,3))
# 回傳一個tensor，其資料和numpy變數是連動的
print(torch.from_numpy(n))
# 回傳一個numpy變數，其資料和tensor是連動的
print(m.numpy())