import torch

# 此处仅以加法为例，其他运算同理
a = torch.rand(2, 3)
b = torch.rand(3)
c = a + b
print("a:", a)
print("b:", b)
print("c:", c)
print("c.shape:", c.shape)

a = torch.rand(2, 1)
b = torch.rand(3)
c = a + b
print("a:", a)
print("b:", b)
print("c:", c)
print("c.shape:", c.shape)

a = torch.rand(2, 1, 1, 3)
b = torch.rand(4, 2, 3)
c = a + b
print("a:", a)
print("b:", b)
print("c:", c)
print("c.shape:", c.shape)