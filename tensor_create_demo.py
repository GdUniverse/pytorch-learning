import torch

# Tensor(data)
a = torch.Tensor([[1, 2, 3], [4, 5, 6]])  # Tensor(data)会默认创建一个浮点型张量，是torch.FloatTensor的简写，只能创建浮点型张量
print(a)
print(a.type())
b = torch.tensor([[1, 2, 3], [4, 5, 6]])  # tensor(data)会自动识别数据类型
print(b)
print(b.type())
c = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float16)  # 创建一个二维张量并指定数据类型为float16(半精度浮点数)
print(c)
print(c.type())

# Tensor(*size)
d = torch.Tensor(2, 3)  # 创建一个2行3列的张量，值为从内存中直接截取出的未初始化数据，没有任何意义
print(d)
print(d.type())
e = torch.Tensor(2, 3, 4)  # 创建一个2层3行4列的张量，即一个三维张量，和numpy的ndarray类似
print(e)
print(e.type())

"""几种特殊的Tensor"""
# ones(*size)
f = torch.ones(2, 3)  # 创建一个2行3列的张量，值全部为1
print(f)
print(f.type())

# zeros(*size)
g = torch.zeros(2, 3)  # 创建一个2行3列的张量，值全部为0
print(g)
print(g.type())

# ones_like(tensor)
h = torch.ones_like(b)  # 创建一个和b形状相同的张量，值全部为1
print(h)
print(h.type())
h = torch.ones_like(a)  # 创建一个和a形状相同的张量，值全部为1
print(h)
print(h.type())

# zeros_like(tensor)
i = torch.zeros_like(b)  # 创建一个和b形状相同的张量，值全部为0
print(i)
print(i.type())
i = torch.zeros_like(a)  # 创建一个和a形状相同的张量，值全部为0
print(i)
print(i.type())

"""随机数张量的创建"""
# rand(*size)
j = torch.rand(2, 3)  # 创建一个2行3列的张量，值为0到1之间的随机数
print(j)
print(j.type())

# normal()
# 创建一个2行3列的张量，值为从正态分布中截取的随机数，张量中每个元素的均值为0，标准差为随机生成的0到1之间的数
k = torch.normal(mean= 0, std= torch.rand(2, 3))
print(k)
print(k.type())
# 创建一个2行3列的张量，值为从正态分布中截取的随机数，张量中每个元素的标准差为随机生成的0到1之间的数，均值为随机生成的0到1之间的数
l = torch.normal(mean= torch.rand(2, 3), std= torch.rand(2, 3))
print(l)
print(l.type())

# uniform()
# uniform()使用需要先创建一个空的张量，然后使用uniform()方法来填充随机数
# 创建一个2行3列的张量，值为从均匀分布中截取的随机数，张量中每个元素的值在0到1之间
m = torch.empty(2, 3).uniform_(0, 1)  # 此处empty()可以用Tensor()代替
print(m)
print(m.type())

"""序列张量的创建"""
# arange(start, end, step)
# 创建一个从0到10之间，步长为2的张量
n = torch.arange(0, 10, 2)  # 类似range()函数，start,end构成左闭右开区间
print(n)
print(n.type())

# linspace(start, end, steps)
# 创建一个从0到1之间，包含5个元素的张量，元素之间的间隔相等
o = torch.linspace(0, 4, steps=5)  # 类似numpy的linspace()函数，start,end构成闭区间
print(o)
print(o.type())

# randperm(n)
# 生成一个0到n-1的随机排列的张量
# 可以用于打乱样本索引
p = torch.randperm(10)  # 创建一个包含0到9之间的随机排列的张量
print(p)
print(p.type())

"""以上操作都可以用numpy实现，可以理解为在numpy输出的数组上套了一层tensor的外衣
很多numpy的操作在tensor中存在，很多tensor的操作在numpy中也存在"""