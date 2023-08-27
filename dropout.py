"""通过dropout进行正则化，随机让部分权重归0，一般放在激活层之后，让模型不过度关注某几个权重造成过拟合"""
import torch
from torch import nn
import d2l

dropout1, dropout2 = 0.2, 0.5


def dropout_layer(X, dropout):
    assert 0 <= dropout <= 1
    # 若为1，则所有元素都被丢弃
    if dropout == 1:
        return torch.zeros_like(X)
    # 若为0，所有元素都被保留
    if dropout == 0:
        return X
    # 在dropout值内，则为0，否则为 h / (1 - p), tensor bool矩阵转float
    mask = (torch.rand(X.shape) > dropout).float()
    return mask * X / (1.0 - dropout)


class Net(nn.Module):
    """层次为 linear relu dropout linear relu dropout linear"""

    def __init__(self, num_inputs, num_outputs, num_hiddens1, num_hiddens2,
                 is_training=True):
        super(Net, self).__init__()
        self.num_inputs = num_inputs
        self.training = is_training
        self.lin1 = nn.Linear(num_inputs, num_hiddens1)
        self.lin2 = nn.Linear(num_hiddens1, num_hiddens2)
        self.lin3 = nn.Linear(num_hiddens2, num_outputs)
        self.relu = nn.ReLU()

    def forward(self, X):
        H1 = self.relu(self.lin1(X.reshape((-1, self.num_inputs))))
        # 只有在训练模型时才使⽤dropout
        if self.training:
            # 在第一个全连接层之后添加一个dropout层
            H1 = dropout_layer(H1, dropout1)
        H2 = self.relu(self.lin2(H1))
        if self.training:
            # 在第二个全连接层之后添加一个dropout层
            H2 = dropout_layer(H2, dropout2)
        out = self.lin3(H2)
        return out


def train():
    num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 784, 10, 256, 256
    net = Net(num_inputs, num_outputs, num_hiddens1, num_hiddens2)
    num_epochs, lr, batch_size = 10, 0.5, 256
    loss = nn.CrossEntropyLoss(reduction='none')
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    trainer = torch.optim.SGD(net.parameters(), lr=lr)
    d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
    d2l.plt.show()

    # pytorch实现
    net = nn.Sequential(nn.Flatten(),
                        nn.Linear(784, 256),
                        nn.ReLU(),
                        # 在第一个全连接层之后添加一个dropout层
                        nn.Dropout(dropout1),
                        nn.Linear(256, 256),
                        nn.ReLU(),
                        # 在第二个全连接层之后添加一个dropout层
                        nn.Dropout(dropout2),
                        nn.Linear(256, 10))

    def init_weights(m):
        if type(m) == nn.Linear:
            nn.init.normal_(m.weight, std=0.01)

    net.apply(init_weights)
    trainer = torch.optim.SGD(net.parameters(), lr=lr)
    d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
    d2l.plt.show()


if __name__ == '__main__':
    train()
