"""LeNet给出了卷积网络的可用性"""
from torch import nn
import d2l


def train():
    # 假设输入1条样本，通道是灰度1，图片形状是28x28
    net = nn.Sequential(  # [1, 1, 28, 28]
        nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.Sigmoid(),  # [1, 6, 28, 28]
        nn.AvgPool2d(kernel_size=2, stride=2),  # [1, 6, 14, 14]
        nn.Conv2d(6, 16, kernel_size=5), nn.Sigmoid(),  # [1, 16, 10, 10]
        nn.AvgPool2d(kernel_size=2, stride=2),  # [1, 16, 5, 5]
        nn.Flatten(),  # [1, 16x5x5]
        nn.Linear(16 * 5 * 5, 120), nn.Sigmoid(),  # [1, 120]
        nn.Linear(120, 84), nn.Sigmoid(),  # [1, 84]
        nn.Linear(84, 10))  # [1, 10]
    batch_size = 256
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)
    lr, num_epochs = 0.9, 10
    d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
    d2l.plt.show()


if __name__ == '__main__':
    train()
