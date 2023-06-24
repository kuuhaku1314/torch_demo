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
    # 这⾥的（1，1）表⽰批量⼤⼩和通道数都是1
    X = X.reshape((1, 1) + X.shape)
    Y = conv2d(X)
    # 省略前两个维度：批量⼤⼩和通道
    return Y.reshape(Y.shape[2:])


def train():
    X = torch.ones((6, 8))
    X[:, 2:6] = 0
    K = torch.tensor([[1.0, -1.0]])
    Y = d2l.corr2d(X, K)
    # 构造一个二维卷积层，它具有1个输出通道和形状为（1，2）的卷积核
    conv2d = nn.Conv2d(1, 1, kernel_size=(1, 2), bias=False)
    # 这个二维卷积层使⽤四维输⼊和输出格式（批量大小、通道、⾼度、宽度），
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
    print(conv2d.weight.data.reshape((1, 2)))
    X = torch.ones((8, 8))
    conv2d = nn.Conv2d(1, 1, kernel_size=(5, 3), padding=(2, 1), stride=(2, 2))
    print(comp_conv2d(conv2d, X).shape)


if __name__ == '__main__':
    train()
