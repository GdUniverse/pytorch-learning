# 机器学习

## 问题分类

预测结果离散$\rightarrow$分类问题

预测结果连续$\rightarrow$回归问题

## 机器学习问题的构成元素

一个完整的机器学习问题通常包含以下元素：

- 样本
- 模型
- 训练
- 测试
- 推理

机器学习并不是进行盲目的推理，而是根据隐式的内容抽取出客观规律，然后用这些规律进行客观的推理和预测。这些规律即是通过样本挖掘出来的，这便是机器学习中的"学习"的体现。

### 样本

样本通常包含两个部分：$\left\{\begin{array}{l}属性\\标签\end{array}\right.$

属性是对样本的描述：比如一个人长头发没有喉结，另一个人短头发有喉结

标签是样本的类别：比如根据以上描述，第一个人为女人，第二个人为男人。`男人`、`女人`就是样本的类别，即标签

从函数的角度理解。输入了一组样本$[x,y]$，$y$与$x$满足一个函数关系$y=f(x)$。$x$即为属性而$y$即为标签

### 模型

`对样本进行建模`，这一步骤即被描述为`抽取模型`

从函数的角度理解。$f$（函数关系）即为模型

### 训练

通过挖掘样本属性的内在联系得到样本的标签，`学习这个模型的过程`即为训练

从函数的角度理解。相当于求解函数$f$，若$f(x)=wx+b$则是求解$w、b$

由于通过提供的样本来求解函数$f$，样本数和未知参数的数目不一定相等，如果用传统消元求解的方法甚至会出现矛盾解，所以采取训练的方法来求解

### 测试

通过训练求解获得参数后，就可以对参数进行测试、评价。测试其实就是对参数的评价、达成`对训练求解获得的模型的性能的评估`

### 推理

在测试模型确定模型性能不错的情况下，如果获得了一组样本只有属性没有标签，就可以通过这个模型进行推理获得模型的标签。所谓推理的过程就是"计算"`标签`的过程

## Tensor(张量)的基本概念

在实际生活中矩阵的$m\times n$是不够用的，需要更多的维度$m\times n\times o \times ···$，于是张量诞生了。它可以创建任意维度的数据

标量是0维的张量、向量是1维的张量、矩阵是2维的张量

在一个完整机器学习任务中，`样本`和`模型`是两个非常重要的概念

样本通常是一组数据(非数据也会被转化为数据)，这些数据会被转化为tensor(张量)表示，用tensor来描述样本

模型可以被分为两类：`有参数模型`、`无参数模型`

| 特性   | 有参数模型（Parametric）     | 无参数模型（Non-parametric）                |
| ---- | --------------------- | ------------------------------------ |
| 参数数量 | 固定（与数据量无关）            | 不固定，随数据量变化                           |
| 假设   | 假设数据满足某种分布（例如线性）      | 不做强假设                                |
| 表达能力 | 可能不足以表达复杂关系           | 灵活，拟合能力强                             |
| 例子   | 线性回归、逻辑回归、神经网络（结构固定时） | k近邻（kNN）、决策树、核密度估计、随机森林、支持向量机（某种意义上） |

样本和模型可以表示为$Y=WX+b$

其中$Y$表示样本标签，$X$表示样本属性(可简称'样本')，$W、b$在模型还没有被求解出来时为变量——以上所有参数在机器学习问题中都会被转化为tensor(用tensor描述)

## Tensor的常用操作(使用Tensor时的基本概念)

- Tensor的类型
- 如何创建Tensor
- Tensor包含哪些属性
- 如何使用Tensor进行算术运算
- Tensor的其他的操作(如：切片、索引、变形)
- Tensor与numpy互相转换

### Tensor的类型

| Data Type                     | PyTorch Alias                          | Tensor Type                  |
|-------------------------------|----------------------------------------|------------------------------|
| 32-bit floating point         | `torch.float32` or `torch.float`      | `torch.*.FloatTensor`        |
| 64-bit floating point         | `torch.float64` or `torch.double`     | `torch.*.DoubleTensor`       |
| 16-bit floating point         | `torch.float16` or `torch.half`       | `torch.*.HalfTensor`         |
| 8-bit integer (unsigned)      | `torch.uint8`                         | `torch.*.ByteTensor`         |
| 8-bit integer (signed)        | `torch.int8`                          | `torch.*.CharTensor`         |
| 16-bit integer (signed)       | `torch.int16` or `torch.short`        | `torch.*.ShortTensor`        |
| 32-bit integer (signed)       | `torch.int32` or `torch.int`          | `torch.*.IntTensor`          |
| 64-bit integer (signed)       | `torch.int64` or `torch.long`         | `torch.*.LongTensor`         |
| Boolean                       | `torch.bool`                          | `torch.*.BoolTensor`         |

### Tensor的创建

#### 常用的函数及其功能

| 函数                     | 功能                           |
|--------------------------|--------------------------------|
| `Tensor(*size)`          | 基础构造函数                   |
| `Tensor(data)`           | 类似 `np.array`                |
| `ones(*size)`            | 全 1 张量 (`Full1Tensor`)      |
| `zeros(*size)`           | 全 0 张量 (`Full0Tensor`)      |
| `eye(*size)`             | 对角线为 1，其他为 0           |
| `arange(s, e, step)`     | 从 `s` 到 `e`，步长为 `step`   |
| `linspace(s, e, steps)`  | 从 `s` 到 `e`，均匀切分成 `steps` 份 |
| `rand/randn(*size)`      | 均匀/标准分布                  |
| `normal(mean, std)/uniform_(from, to)` | 正态分布/均匀分布         |
| `randperm(m)`            | 随机排列                       |

Tensor的创建可以结合、类比numpy进行理解

#### Tensor的创建编程实例

- [Tensor创建实例(py文件)](tensor_create_demo.py)
- [Tensor创建实例(ipynb文件)](tensor_create_demo.ipynb)

### Tensor的属性

- 每一个 Tensor 有 `torch.dtype`、`torch.device`、`torch.layout` 三种属性。
- `torch.dtype` 标识了 `torch.Tensor` 对象的数据类型
- `torch.device` 标识了 `torch.Tensor` 对象在创建之后所存储在的设备名称。
  - 设备可以分为 CPU 和 GPU。
  - CPU: `torch.device("cpu")`
  - GPU:
    - 单块GPU:`torch.device("cuda")`
    - 多块GPU:`torch.device("cuda:0")`
- `torch.layout` 表示 `torch.Tensor` 内存布局的对象。
  - [Tensor的创建编程实例](#tensor的创建编程实例)展示的是 `torch.Tensor` 的内存布局为 `torch.strided`，称为稠密张量
  - 除了 `torch.strided`(稠密张量) 外，还有稀疏张量
  
Tensor创建语句示列：`torch.tensor(data, dtype=torch.float32, device=torch.device("cpu"))`，默认为稠密张量

### Tensor的属性——稀疏张量

- `torch.sparse_coo_tensor`是最常用的稀疏张量类型，适用于存储大部分元素为零的张量。它使用坐标格式(COO)来存储非零元素的位置和数值。
- 张量中0元素越多越稀疏，全为0则最稀疏。
- 用线性代数理解即为低秩矩阵。
- 使用稀疏张量的意义：
  - 对于有参数模型，如果导入的参数有很多为0的参数，可以将0消去简化模型。
  - 对于有大量0元素的矩阵，稀疏张量只存储非零元素，减少了内存占用。(如果使用稠密张量存储，内存占用会很大)
    - 如一个100x100的矩阵，只有一个参数非零，其他参数都为0，稠密张量会占用10000个元素的内存，而稀疏张量只存储一个非零元素和其相应的坐标
    - 稀疏张量创建:

    ```python
    import torch

    # 创建稀疏张量
    indices = torch.tensor([[0, 1, 2], [2, 0, 1]])  # 坐标值 表示3个非零元素的位置(0, 1), (2, 0), (2, 1)
    values = torch.tensor([3, 4, 5])  # 非零元素的值 和坐标值一一对应
    sparse_tensor = torch.sparse_coo_tensor(indices, values, (3, 3))

    print(sparse_tensor)
    ```
