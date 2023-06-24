"""softmax是把预测值归一化到总和为1的概率分布的方式，一般和交叉熵损失一起使用，最小化交叉熵损失也就是最大化似然"""
import torch
import d2l
from torch import nn


def train():
    batch_size = 256
    # 数据集形状为(60000, 1) 含义分别为样本数量，图像或label
    # 若是label 使用[0][1]获取分类，值为标签
    # 若是图像，使用[0][0]获取图像，图像数据为2个维度，高度，宽度，值为灰度
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=None)
    num_inputs = 784
    num_outputs = 10
    W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
    b = torch.zeros(num_outputs, requires_grad=True)

    def net(X):
        # (1，784) X (784，10) + (10)
        return d2l.softmax(torch.matmul(X.reshape((-1, W.shape[0])), W) + b)

    lr = 0.1

    def updater(_batch_size):
        return d2l.sgd([W, b], lr, _batch_size)

    num_epochs = 10
    d2l.train_ch3(net, train_iter, test_iter, d2l.cross_entropy, num_epochs, updater)
    d2l.plt.show()

    d2l.predict_ch3(net, test_iter)
    d2l.plt.show()

    # 使用pytorch库实现
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    # PyTorch不会隐式地调整输⼊的形状。因此，
    # 我们在线性层前定义了展平层（flatten），来调整⽹络输⼊的形状
    net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))

    def init_weights(m):
        if type(m) == nn.Linear:
            nn.init.normal_(m.weight, std=0.01)
            m.bias.data.fill_(0)

    # 对每个层施加一个函数
    net.apply(init_weights)
    # 使用none则loss会组合成一个向量，而在train里使用了mean函数聚合为标量再反向传播
    loss = nn.CrossEntropyLoss(reduction='none')
    trainer = torch.optim.SGD(net.parameters(), lr=0.1)
    num_epochs = 10
    d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
    d2l.plt.show()


if __name__ == '__main__':
    train()
