import torch

# 指定设备创建张量
dev = torch.device("cuda:0")
# torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 根据设备情况自动判断使用哪个设备

a = torch.tensor([[0, 1, 2], [2, 0, 1]])  # 创建张量，不指定设备的情况下默认为 CPU
print(a)
print(a.type())
b = torch.tensor([[0, 1, 2], [2, 0, 1]], device=dev)  # 创建张量并指定设备
print(b)
print(b.type())
b = torch.tensor([[0, 1, 2], [2, 0, 1]], dtype=torch.float32, device="cuda")  # 创建张量并指定设备
print(b)
print(b.type())

# 稀疏张量创建
i = torch.tensor([[0, 1, 2], [0, 1, 2]])  # 稀疏张量的索引(坐标)(行, 列)
v = torch.tensor([1, 2, 3])  # 稀疏张量的值
c = torch.sparse_coo_tensor(i, v, (4, 4), dtype=torch.float32, device=dev)  # 创建稀疏张量，并指定数据类型和设备
print(c)
print(c.type())  # 打印稀疏张量类型
print(c.to_dense())  # 转换为稠密张量
