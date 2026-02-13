"""卷积网络一般用来图像处理，可以学习到图像的空间特征，并有一定的平移不变性"""
import torch
from torch import nn
import d2l


class Conv2D(nn.Module):

    def __init__(self, kernel_size):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(kernel_size))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return d2l.corr2d(x, self.weight) + self.bias


def comp_conv2d(conv2d, X):
    # 这里的(1，1)表⽰批量大小和通道数都是1
    X = X.reshape((1, 1) + X.shape)
    Y = conv2d(X)
    # 省略前两个维度：批量大小和通道
    return Y.reshape(Y.shape[2:])


def corr2d_multi_in(X, K):
    # X形状是 channel h w， K形状是 channel h w，结果是h w
    # 先遍历“X”和“K”的第0个维度(通道维度)，再把它们加在一起，多输入通道单输出通道
    # 举个例子X=[[[1, 1], [1, 1]], [[2, 2], [2, 2]]], K=[[[1, 1], [1, 1]], [[1, 1], [1, 1]]], 则结果=[[12]]
    return sum(d2l.corr2d(x, k) for x, k in zip(X, K))


def corr2d_multi_in_out(X, K):
    # X形状是 channel h w， K形状是 o channel h w，结果是o h w, 因为在corr2d_multi_in中把channel维度消除了
    # 迭代“K”的第0个维度，每次都对输⼊“X”执行互相关运算。
    # 最后将所有结果都叠加在一起
    # 可以这样理解，假设X = channel h w和K = channel h w
    # 每个二维卷积核h w对应了一个X的通道，每个二维核只响应对应通道的输入，也就是每个二维核只对输入数据对应单通道的特征有反应
    # 在corr2d_multi_in中，把每个二维核和对应通道的X卷积结果叠加起来，则构成了对所有特征反应结果的和
    # 若输出为多通道，也就是K = o channel h w，则相当于有多个三维卷积核，每个三维卷积核都采样整体输入数据的某个特征
    return torch.stack([corr2d_multi_in(X, k) for k in K], 0)


def corr2d_multi_in_out_1x1(X, K):
    """用矩阵乘法实现1x1卷积核"""
    c_i, h, w = X.shape
    c_o = K.shape[0]
    # 把像素展平成行，输入通道展平成列
    X = X.reshape((c_i, h * w))
    # 1x1卷积核不对像素产生影响，关键是对各通道的响应之和
    K = K.reshape((c_o, c_i))
    # 全连接层中的矩阵乘法
    Y = torch.matmul(K, X)
    # 可以看出1x1卷积核原样进行了通道->通道的映射，对形状没有改变
    return Y.reshape((c_o, h, w))


def pool2d(X, pool_size, mode='max'):
    """汇聚层，将周围元素的特征汇聚到一起，降低卷积层对位置的敏感性"""
    # 原理是图像产生了平移，在汇聚层汇聚特征时，依然能捕捉到平移后的特征
    # 注意，汇聚层不像卷积核会把多个通道结果加在一起，而是在每个通道进行汇聚，前后维度不变
    # max模式适合非平滑图像，mean适合平滑图像
    p_h, p_w = pool_size
    Y = torch.zeros((X.shape[0] - p_h + 1, X.shape[1] - p_w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == 'max':
                Y[i, j] = X[i: i + p_h, j: j + p_w].max()
            elif mode == 'avg':
                Y[i, j] = X[i: i + p_h, j: j + p_w].mean()
    return Y


def train():
    X = torch.ones((6, 8))
    X[:, 2:6] = 0
    K = torch.tensor([[1.0, -1.0]])
    Y = d2l.corr2d(X, K)
    # 构造一个二维卷积层，它具有1个输出通道和形状为(1，2)的卷积核
    conv2d = nn.Conv2d(1, 1, kernel_size=(1, 2), bias=False)
    # 这个二维卷积层使⽤四维输⼊和输出格式(批量大小、通道、⾼度、宽度)，
    # 其中批量大小和通道数都为1
    X = X.reshape((1, 1, 6, 8))
    Y = Y.reshape((1, 1, 6, 7))
    lr = 3e-2  # 学习率
    for i in range(10):
        conv2d.zero_grad()
        Y_hat = conv2d(X)
        l = (Y_hat - Y) ** 2
        l.sum().backward()
        # 迭代卷积核
        conv2d.weight.data[:] -= lr * conv2d.weight.grad
        if (i + 1) % 2 == 0:
            print(f'epoch {i + 1}, loss {l.sum():.3f}')
    # 打印出推断出的卷积核
    print(conv2d.weight.data.reshape((1, 2)))
    # 下面演示经过卷积核后的形状变化 8x8 -> 4x4
    X = torch.ones((8, 8))
    conv2d = nn.Conv2d(1, 1, kernel_size=(5, 3), padding=(2, 1), stride=(2, 2))
    print(comp_conv2d(conv2d, X).shape)


if __name__ == '__main__':
    train()
