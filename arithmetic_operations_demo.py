import torch

# add
a = torch.randn(2, 3)
b = torch.randn(2, 3)
print("a:", a)
print("b:", b)
print("a + b:", a + b)
print("a.add(b):", a.add(b))
print("torch.add(a, b):", torch.add(a, b))
print("a.add_(b):", a.add_(b))
print("a after a.add_(b):", a)

# sub
print("a - b:", a - b)
print("a.sub(b):", a.sub(b))
print("torch.sub(a, b):", torch.sub(a, b))
print("a.sub_(b):", a.sub_(b))
print("a after a.sub_(b):", a)

# mul
print("a * b:", a * b)
print("a.mul(b):", a.mul(b))
print("torch.mul(a, b):", torch.mul(a, b))
print("a.mul_(b):", a.mul_(b))
print("a after a.mul_(b):", a)

# div
print("a / b:", a / b)
print("a.div(b):", a.div(b))
print("torch.div(a, b):", torch.div(a, b))
print("a.div_(b):", a.div_(b))
print("a after a.div_(b):", a)

# matmul
a = torch.randn(2, 3)
b = torch.randn(3, 2)
print("a:", a)
print("b:", b)
print("a @ b:", a @ b)
print("torch.matmul(a, b):", torch.matmul(a, b))
print("torch.mm(a, b):", torch.mm(a, b))
print("a.matmul(b):", a.matmul(b))
print("a.mm(b):", a.mm(b))

# 高维Tensor运算
a = torch.randn(2, 3, 4)
b = torch.randn(2, 4, 5)
print("torch.matmul(a, b).shape:", torch.matmul(a, b).shape)
print("a.matmul(b).shape:", a.matmul(b).shape)

# pow
a = torch.tensor([1, 2])
print("torch.pow(a, 3):", torch.pow(a, 3))
print("a.pow(3):", a.pow(3))
print("a ** 3:", a ** 3)
print("a.pow_(3):", a.pow_(3))
print("a after a.pow_(3):", a)

# exp
a = torch.tensor([1, 2], dtype=torch.float32)  # 对于自然指数运算，输入需要是浮点数类型
print("a.type():", a.type())
print("torch.exp(a):", torch.exp(a))
print("torch.exp_(a):", torch.exp_(a))
print("a after torch.exp_(a):", a)
print("a.exp():", a.exp())
print("a.exp_():", a.exp_())
print("a after a.exp_():", a)

# log
# 此处仅展示自然对数运算，其他对数运算可以使用`torch.log10`或`torch.log2`
a = torch.tensor([10, 2], dtype=torch.float32)
print("torch.log(a):", torch.log(a))
print("torch.log_(a):", torch.log_(a))
print("a after torch.log_(a):", a)
print("a.log():", a.log())
print("a.log_():", a.log_())
print("a after a.log_():", a)

# sqrt
a = torch.tensor([4, 9], dtype=torch.float32)  # 对于平方根运算，输入需要是浮点数类型
print("torch.sqrt(a):", torch.sqrt(a))
print("torch.sqrt_(a):", torch.sqrt_(a))
print("a after torch.sqrt_(a):", a)
print("a.sqrt():", a.sqrt())
print("a.sqrt_():", a.sqrt_())
print("a after a.sqrt_():", a)

"""不建议使用原地运算，原地运算会改变原始数据，可能会导致后续计算出错。"""