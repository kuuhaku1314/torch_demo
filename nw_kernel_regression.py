"""Nadaraya-Watson核回归，早期的注意力机制范例"""
from torch import nn
import d2l
import torch


def train():
    attention_weights = torch.eye(10).reshape((1, 1, 10, 10))
    d2l.show_heatmaps(attention_weights, xlabel='Keys', ylabel='Queries')
    d2l.plt.show()

    n_train = 50
    x_train, _ = torch.sort(torch.rand(n_train) * 5)

    def f(x):
        return 2 * torch.sin(x) + x ** 0.8

    y_train = f(x_train) + torch.normal(0.0, 0.5, (n_train,))
    x_test = torch.arange(0, 5, 0.1)
    y_truth = f(x_test)
    n_test = len(x_test)

    def plot_kernel_reg(y_hat):
        d2l.plot(x_test, [y_truth, y_hat], 'x', 'y', legend=['Truth', 'Pred'],
                 xlim=[0, 5], ylim=[-1, 5])
        d2l.plt.plot(x_train, y_train, 'o', alpha=0.5)

    y_hat = torch.repeat_interleave(y_train.mean(), n_test)
    plot_kernel_reg(y_hat)
    d2l.plt.show()

    # 下面把训练的x视为key，测试的x视为query，若key和query的距离越远，则权重越低。
    # 使用高斯核来计算权重，约束权重总和为1，高斯核可以化简为softmax

    # X_repeat的形状:(n_test,n_train),
    # 每一行都包含着相同的测试输入（例如：同样的查询）
    X_repeat = x_test.repeat_interleave(n_train).reshape((-1, n_train))
    # x_train包含着键。attention_weights的形状：(n_test,n_train)，每行权重和为1
    # 每一行都包含着要在给定的每个查询的值（y_train）之间分配的注意力权重
    attention_weights = nn.functional.softmax(-(X_repeat - x_train) ** 2 / 2, dim=1)
    # 通过高斯核对key-query建模，给出的test_x离对应的train_x越近则获得越高的注意力，也就是权重越高
    # y_hat的每个元素都是值的加权平均值，其中的权重是注意力权重
    # 矩阵和向量的乘法，相当于x_test每一行和y_train的点积
    y_hat = torch.matmul(attention_weights, y_train)
    plot_kernel_reg(y_hat)
    d2l.plt.show()

    # 画出矩阵热力图，很显然越接近对角线热力越高
    d2l.show_heatmaps(attention_weights.unsqueeze(0).unsqueeze(0),
                      xlabel='Sorted training inputs',
                      ylabel='Sorted testing inputs')
    d2l.plt.show()

    # 带有可学习参数的模型 在(x1-x2)外层加上一个权重w，构成(x1-x2)w
    # 从注意力的角度来看，分配给每个值的注意力权重取决于将值所对应的键和查询作为输入的函数
    class NWKernelRegression(nn.Module):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.w = nn.Parameter(torch.rand((1,), requires_grad=True))

        def forward(self, queries, keys, values):
            # queries和attention_weights的形状为(查询个数，“键－值”对个数)
            queries = queries.repeat_interleave(keys.shape[1]).reshape((-1, keys.shape[1]))
            self.attention_weights = nn.functional.softmax(
                -((queries - keys) * self.w) ** 2 / 2, dim=1)
            # values的形状为(查询个数，“键－值”对个数)
            return torch.bmm(self.attention_weights.unsqueeze(1),
                             values.unsqueeze(-1)).reshape(-1)

    # X_tile的形状:(n_train，n_train)，每一行都包含着相同的训练输入
    X_tile = x_train.repeat((n_train, 1))
    # Y_tile的形状:(n_train，n_train)，每一行都包含着相同的训练输出
    Y_tile = y_train.repeat((n_train, 1))

    # 任何一个训练样本的输入都会和除自己以外的所有训练样本的“键－值”对进行计算，应该是避免过拟合
    # keys的形状:(n_train，n_train-1)，每行少了一个单位矩阵位置的值
    keys = X_tile[(1 - torch.eye(n_train)).type(torch.bool)].reshape((n_train, -1))
    # values的形状:(n_train，n_train-1)，每行少了一个单位矩阵位置的值
    values = Y_tile[(1 - torch.eye(n_train)).type(torch.bool)].reshape((n_train, -1))

    net = NWKernelRegression()
    loss = nn.MSELoss(reduction='none')
    trainer = torch.optim.SGD(net.parameters(), lr=0.5)
    animator = d2l.Animator(xlabel='epoch', ylabel='loss', xlim=[1, 5])

    for epoch in range(5):
        trainer.zero_grad()
        l = loss(net(x_train, keys, values), y_train)
        l.sum().backward()
        trainer.step()
        print(f'epoch {epoch + 1}, loss {float(l.sum()):.6f}')
        animator.add(epoch + 1, float(l.sum()))
    # keys的形状:(n_test，n_train)，每一行包含着相同的训练输入（例如，相同的键）
    keys = x_train.repeat((n_test, 1))
    # value的形状:(n_test，n_train)
    values = y_train.repeat((n_test, 1))
    y_hat = net(x_test, keys, values).unsqueeze(1).detach()
    plot_kernel_reg(y_hat)
    d2l.plt.show()
    d2l.show_heatmaps(net.attention_weights.unsqueeze(0).unsqueeze(0).detach(),
                      xlabel='Sorted training inputs',
                      ylabel='Sorted testing inputs')
    d2l.plt.show()


if __name__ == '__main__':
    train()
