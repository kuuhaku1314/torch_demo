"""VGG把多个层组合成一个块，当做网络的基本元素使用"""
import torch
from torch import nn
import d2l


def vgg_block(num_convs, in_channels, out_channels):
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels, out_channels,
                                kernel_size=(3, 3), padding=1))
        layers.append(nn.ReLU())
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)


def vgg(conv_arch):
    conv_blks = []
    in_channels = 1
    out_channels = None
    # 卷积层部分
    for (num_convs, _out_channels) in conv_arch:
        conv_blks.append(vgg_block(num_convs, in_channels, _out_channels))
        in_channels = _out_channels
        out_channels = _out_channels

    return nn.Sequential(
        *conv_blks, nn.Flatten(),
        # 全连接层部分
        nn.Linear(out_channels * 7 * 7, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 10))


def train():
    conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))
    net = vgg(conv_arch)
    d2l.print_net(net, torch.randn(size=(1, 1, 224, 224)))

    ratio = 4
    small_conv_arch = [(pair[0], pair[1] // ratio) for pair in conv_arch]
    net = vgg(small_conv_arch)
    lr, num_epochs, batch_size = 0.05, 10, 128
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
    d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
    d2l.plt.show()


if __name__ == '__main__':
    train()
