"""NiN从另一种角度设计了神经网络的层，不在最后使用全连接层输出预测值，而是把卷积层的通道直接当做预测值"""
import torch
from torch import nn
import d2l


# NiN层舍弃了线性层，把每个通道作为预测值
def nin_block(in_channels, out_channels, kernel_size, strides, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, strides, padding),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=(1, 1)), nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=(1, 1)), nn.ReLU())


def train():
    net = nn.Sequential(
        nin_block(1, 96, kernel_size=(11, 11), strides=4, padding=0),
        nn.MaxPool2d(3, stride=2),
        nin_block(96, 256, kernel_size=(5, 5), strides=1, padding=2),
        nn.MaxPool2d(3, stride=2),
        nin_block(256, 384, kernel_size=(3, 3), strides=1, padding=1),
        nn.MaxPool2d(3, stride=2),
        nn.Dropout(0.5),
        # 标签类别数是10
        nin_block(384, 10, kernel_size=(3, 3), strides=1, padding=1),
        nn.AdaptiveAvgPool2d((1, 1)),
        # 将四维的输出转成二维的输出，其形状为(批量大小,10)
        nn.Flatten())
    X = torch.rand(size=(1, 1, 224, 224))
    d2l.print_net(net, X)
    lr, num_epochs, batch_size = 0.01, 10, 128
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
    d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
    d2l.plt.show()


if __name__ == '__main__':
    train()
