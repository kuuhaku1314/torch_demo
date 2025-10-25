import numpy as np

# 声明一个ndarray数组
a = np.array([[1, 2], [3, 4]])
# 声明一个值为0，长度为100的ndarray数组
np.zeros(100)
# 声明一个长度为10的ndarray数组，第一个值为0
b = np.arange(2)
# 给0-2位置索引元素赋值
b[0:2] = a[0]
# 若是向量dot向量，则是向量点积得到标量，
# 若是向量dot矩阵，则会把向量提升为矩阵后再降维，比如20 dot 20x10 -> 1x20 dot 20x10 -> 1x10 -> 10，和pytorch matmul表现一致
# 若是矩阵dot矩阵，执行矩阵乘法，如5x3 dot 3x5 -> 5x5
np.dot(a, b)
# 矩阵转置
np.transpose(a)
# 生成均值为5，标准差为0.1，形状为5x2的正态分布数组
a = np.random.normal(loc=5, scale=0.1, size=(5, 2))
# 打乱数组元素
np.random.shuffle(a)
a = np.array([[1], [2], [3], [4]])
# a为[[1], [2], [3], [4]]，这里b为[0, 1]，结果形状是4x2，也就是a进行列广播，b进行行广播
np.power(a, np.arange(2).reshape(1, -1))
# [:, 0]是对二维数组进行切片，2: 说明对第3行到最后一行进行切片，0说明对第0列进行切片
a = a[2:, 0]
