"""多项式回归，把每项预处理后就成了线性回归问题，这里通过使用选取不同的特征，展现了特征太少则欠拟合模型，特征太多则过拟合模型"""
import math
import numpy as np
import torch
from torch import nn
import d2l


def train():
    max_degree = 20  # 多项式的最⼤阶数
    n_train, n_test = 100, 100  # 训练和测试数据集大小
    true_w = np.zeros(max_degree)  # 分配⼤量的空间
    true_w[0:4] = np.array([5, 1.2, -3.4, 5.6])
    features = np.random.normal(size=(n_train + n_test, 1))
    np.random.shuffle(features)
    # 200x1 power 1x20 -> 200x20，生成了200组样本值
    poly_features = np.power(features, np.arange(max_degree).reshape(1, -1))
    for i in range(max_degree):
        poly_features[:, i] /= math.gamma(i + 1)  # gamma(n)=(n-1)!
    # labels的维度:(n_train+n_test, 1) 计算标签，也就是多项式w(1+x+x^2/2!+x^3/3!...)的值
    labels = np.dot(poly_features, true_w)
    # 添加高斯噪声
    labels += np.random.normal(scale=0.1, size=labels.shape)
    true_w, poly_features, labels = [torch.tensor(x, dtype=torch.float32) for x in
                                     [true_w, poly_features, labels]]
    # 从多项式特征中选择前4个维度，即1,x,x^2/2!,x^3/3!
    _train(poly_features[:n_train, :4], poly_features[n_train:, :4],
           labels[:n_train], labels[n_train:], num_epochs=1500)
    d2l.plt.show()
    # 从多项式特征中选择前2个维度，即1和x
    _train(poly_features[:n_train, :2], poly_features[n_train:, :2],
           labels[:n_train], labels[n_train:])
    # 多项式本身有4项特征，只选2项特征欠拟合严重
    d2l.plt.show()
    # 所有特征
    _train(poly_features[:n_train, :], poly_features[n_train:, :],
           labels[:n_train], labels[n_train:], num_epochs=1500)
    # 多项式本身有4项特征，选20项特征，存在过拟合问题
    d2l.plt.show()


def _train(train_features, test_features, train_labels, test_labels,
           num_epochs=400):
    loss = nn.MSELoss(reduction='none')
    input_shape = train_features.shape[-1]
    # 不设置偏置，因为在多项式中存在指数为0的x项
    net = nn.Sequential(nn.Linear(input_shape, 1, bias=False))
    batch_size = min(10, train_labels.shape[0])
    train_iter = d2l.load_array((train_features, train_labels.reshape(-1, 1)),
                                batch_size)
    test_iter = d2l.load_array((test_features, test_labels.reshape(-1, 1)),
                               batch_size, is_train=False)
    trainer = torch.optim.SGD(net.parameters(), lr=0.01)
    animator = d2l.Animator(xlabel='epoch', ylabel='loss', yscale='log',
                            xlim=[1, num_epochs], ylim=[1e-3, 1e2],
                            legend=['train loss', 'test loss'])
    for epoch in range(num_epochs):
        d2l.train_epoch_ch3(net, train_iter, loss, trainer)
        if epoch == 0 or (epoch + 1) % 20 == 0:
            animator.add(epoch + 1, (d2l.evaluate_loss(net, train_iter, loss),
                                     d2l.evaluate_loss(net, test_iter, loss)))
    print('weight:', net[0].weight.data.numpy())


if __name__ == '__main__':
    train()
