"""时间序列回归，根据已知序列，预测下一个时间点的数据"""
import torch
from matplotlib import pyplot as plt
import d2l
from torch import nn


def train():
    # 画出sin(x)的图像
    T = 1000
    time = torch.arange(1, T + 1, dtype=torch.float32)
    x = torch.sin(0.01 * time) + torch.normal(0, 0.2, (T,))
    plt.plot(time.numpy(), x.numpy())
    plt.xlabel('time')
    plt.ylabel('x')
    plt.xlim([1, 1000])
    plt.show()

    tau = 4
    features = torch.ones(T - tau, tau)
    # 每4个时间点的值构成一个feature, 下一个时间点的值构成label
    # 也就是这个模型是根据已有的4个时间点的数据预测下一个时间点的数据
    for i in range(tau):
        features[:, i] = x[i: T - tau + i]
    print(features.shape)
    labels = x[tau:].reshape((-1, 1))
    batch_size, n_train = 16, 600
    # 只有前n_train个样本用于训练
    train_iter = d2l.load_array((features[:n_train], labels[:n_train]),
                                batch_size, is_train=True)

    # 初始化网络权重的函数
    def init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)

    # 一个简单的多层感知机
    net = nn.Sequential(nn.Linear(4, 10),
                        nn.ReLU(),
                        nn.Linear(10, 1))

    net.apply(init_weights)
    loss = nn.MSELoss(reduction='none')

    def _train(net, train_iter, loss, epochs, lr):
        trainer = torch.optim.Adam(net.parameters(), lr)
        for epoch in range(epochs):
            for X, y in train_iter:
                trainer.zero_grad()
                l = loss(net(X), y)
                l.sum().backward()
                trainer.step()
            print(f'epoch {epoch + 1}, 'f'loss: {d2l.evaluate_loss(net, train_iter, loss):f}')

    _train(net, train_iter, loss, 10, 0.1)
    # 展示预测的结果
    onestep_preds = net(features)
    d2l.plot([time, time[tau:]],
             [x.detach().numpy(), onestep_preds.detach().numpy()], 'time',
             'x', legend=['data', '1-step preds'], xlim=[1, 1000],
             figsize=(6, 3))
    d2l.plt.show()

    # 下面展示只有前半部分数据，通过模型预测后续数据的情况，也就是说，后续数据是通过预测出的数据进行预测
    multistep_preds = torch.zeros(T)
    multistep_preds[: n_train + tau] = x[: n_train + tau]
    for i in range(n_train + tau, T):
        multistep_preds[i] = net(
            multistep_preds[i - tau:i].reshape((1, -1)))
    d2l.plot([time, time[tau:], time[n_train + tau:]],
             [x.detach().numpy(), onestep_preds.detach().numpy(),
              multistep_preds[n_train + tau:].detach().numpy()], 'time',
             'x', legend=['data', '1-step preds', 'multistep preds'],
             xlim=[1, 1000], figsize=(6, 3))
    d2l.plt.show()

    max_steps = 64
    features = torch.zeros((T - tau - max_steps + 1, tau + max_steps))
    # feature前4列是真实数据
    for i in range(tau):
        features[:, i] = x[i: i + T - tau - max_steps + 1]
    # 把64步预测数据拼在后64列，也就是说，第5列是预测后1步(用了0步预测数据)，第6列是预测后2步(用了1步预测数据)...第68列是预测后64步
    for i in range(tau, tau + max_steps):
        features[:, i] = net(features[:, i - tau:i]).reshape(-1)
    # 画出图像，可见预测数据离真实数据时间离得越远，预测结果就越不准确，也就是越预测越不靠谱
    steps = (1, 4, 16, 64)
    d2l.plot([time[tau + i - 1: T - max_steps + i] for i in steps],
             [features[:, tau + i - 1].detach().numpy() for i in steps], 'time', 'x',
             legend=[f'{i}-step preds' for i in steps], xlim=[5, 1000],
             figsize=(6, 3))
    d2l.plt.show()


if __name__ == '__main__':
    train()
