"""线性回归，回归是一类基本问题，预测输入下输出值的大小，最小化均方损失等同于正态分布下最大似然"""
import d2l
import torch
import random
from torch import nn


def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    # 这些样本是随机读取的，没有特定的顺序
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(
            indices[i: min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]


def train():
    true_w = torch.tensor([2, -3.4])
    true_b = 4.2
    features, labels = d2l.synthetic_data(true_w, true_b, 10000)
    d2l.set_figsize()
    d2l.plt.scatter(features[:, 1].detach().numpy(), labels.detach().numpy(), 1)
    d2l.plt.show()
    w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    batch_size = 100
    lr = 0.06
    num_epochs = 50
    net = d2l.linreg
    loss = d2l.squared_loss
    sgd = d2l.sgd
    for epoch in range(num_epochs):
        for X, y in data_iter(batch_size, features, labels):
            l = loss(net(X, w, b), y)  # X和y的小批量损失
            # 因为l形状是(batch_size,1)，⽽不是一个标量。l中的所有元素被加到一起，
            # 并以此计算关于[w,b]的梯度
            l.sum().backward()
            sgd([w, b], lr, batch_size)  # 使⽤参数的梯度更新参数
        with torch.no_grad():
            train_l = loss(net(features, w, b), labels)
            print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')
    print(f'w的估计误差: {true_w - w.reshape(true_w.shape).detach()}')
    print(f'b的估计误差: {true_b - b.detach()}')

    # 使用pytorch库实现
    net = nn.Sequential(nn.Linear(2, 1))
    net[0].weight.data.normal_(0, 0.01)
    net[0].bias.data.fill_(0)
    loss = nn.MSELoss(reduction='mean')
    trainer = torch.optim.SGD(net.parameters(), lr=0.03)
    for epoch in range(num_epochs):
        for X, y in d2l.load_array([features, labels], batch_size, is_train=True):
            # 上一轮梯度清空
            trainer.zero_grad()
            # 计算损失
            l = loss(net(X), y)
            # 反向传播
            l.backward()
            # 根据这一轮的梯度优化参数值
            trainer.step()
        l = loss(net(features), labels)
        print(f'epoch {epoch + 1}, loss {l:f}')


if __name__ == '__main__':
    train()
