import torch

dev = torch.device("cpu")
# torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 根据设备情况自动判断使用哪个设备

a = torch.tensor([[0, 1, 2], [2, 0, 1]], device=dev)  # 创建张量并指定设备
print(a)
print(a.type())
