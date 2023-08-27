"""ResNet残差网络，输入可以通过层间的残余连接更快地向前传播，且每加一个残差层，原先层可表示的函数集都是新网络可表示函数的子集
利用了f(x) = (f(x) - x) + x，其中(f(x) - x)和x作为了残差层输入到输出间的两个并行块"""
import torch
from torch import nn
import d2l


def resnet_block(input_channels, num_channels, num_residuals,
                 first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(d2l.Residual(input_channels, num_channels, strides=2))
        else:
            blk.append(d2l.Residual(num_channels, num_channels))
    return blk


def train():
    b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=3),
                       nn.BatchNorm2d(64), nn.ReLU(),
                       nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
    b3 = nn.Sequential(*resnet_block(64, 128, 2))
    b4 = nn.Sequential(*resnet_block(128, 256, 2))
    b5 = nn.Sequential(*resnet_block(256, 512, 2))
    net = nn.Sequential(b1, b2, b3, b4, b5,
                        nn.AdaptiveAvgPool2d((1, 1)),
                        nn.Flatten(), nn.Linear(512, 10))
    d2l.print_net(net, torch.rand(size=(1, 1, 224, 224)))
    lr, num_epochs, batch_size = 0.05, 10, 256
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)
    d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
    d2l.plt.show()


if __name__ == '__main__':
    train()
