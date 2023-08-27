import d2l
import torch
from torch import nn


def train():
    # 每次迭代加载32个样本，每个样本含有长度为35的序列信息
    batch_size, num_steps = 32, 35
    train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps, max_tokens=20000)

    def get_params(vocab_size, num_hiddens, device):
        num_inputs = num_outputs = vocab_size

        def normal(shape):
            return torch.randn(size=shape, device=device) * 0.01

        # 这是一个有一个隐藏层的rnn，假设输入数据为32条，有28个词元，隐藏层有512个参数，输出层有32个参数

        # 隐藏层参数
        # (28X512)
        W_xh = normal((num_inputs, num_hiddens))
        # (512X512)
        W_hh = normal((num_hiddens, num_hiddens))
        # (512)
        b_h = torch.zeros(num_hiddens, device=device)
        # 输出层参数
        # (512X28)
        W_hq = normal((num_hiddens, num_outputs))
        # (28)
        b_q = torch.zeros(num_outputs, device=device)
        # 附加梯度
        params = [W_xh, W_hh, b_h, W_hq, b_q]
        for param in params:
            param.requires_grad_(True)
        return params

    def init_rnn_state(batch_size, num_hiddens, device):
        # (32, 512)
        return (torch.zeros((batch_size, num_hiddens), device=device),)

    def rnn(inputs, state, params):
        # inputs的形状：(时间步数量，批量大小，词表大小(独热向量)) (35X32X28)
        # (28X512), (512X512), (512X28), (512X28), (28)
        W_xh, W_hh, b_h, W_hq, b_q = params
        H, = state  # 形状(批量大小，词表大小) (32, 512)
        outputs = []
        # X的形状：(批量大小，词表大小) (32X28)
        # 这里处理有点特别，每次是处理一个时间步的批量数据
        for X in inputs:
            # (32X28) X (28X512) + (32, 512) X (512X512)-> (32X512) 中间隐藏层和隐变量输出之和
            # 需要注意的是，每个批次中对应位置的样本向量，下一个批次里对应位置的样本向量是连续的
            # 所以隐变量可以学到连续的序列，而不只是单单长度为35的句子，注意若每个批次越大，则单个连续序列越小
            # 公式为batch_size * series_size = text_size
            H = torch.tanh(torch.mm(X, W_xh) + torch.mm(H, W_hh) + b_h)
            # Y的形状：(批量大小，词表大小) (32X512) X (512X28) -> (32X28)
            Y = torch.mm(H, W_hq) + b_q
            outputs.append(Y)
        # 总结上面计算，这里有35个真实时间步数据，假设范围是[1, 35]，接下来根据隐变量和真实的第1个时间步数据，推断出第2个时间步数据
        # 逐次进行，则会推断出[2, 36]时间步数据，最后在外层用交叉熵方法，用预估的时间步数据和真实的[2, 36]时间步数据计算损失
        # 返回输出，输出是估计值(时间步数量X批量大小，词表大小)及隐变量
        # (1120, 28), (32, 512)
        return torch.cat(outputs, dim=0), (H,)

    num_hiddens = 512
    net = d2l.RNNModelScratch(len(vocab), num_hiddens, d2l.try_gpu(), get_params,
                              init_rnn_state, rnn)

    num_epochs, lr = 500, 1
    d2l.train_ch8(net, train_iter, vocab, lr, num_epochs, d2l.try_gpu())
    d2l.plt.show()

    # pytorch实现
    batch_size, num_steps = 32, 35
    train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)
    num_hiddens = 256
    rnn_layer = nn.RNN(len(vocab), num_hiddens)
    num_epochs, lr = 500, 1
    device = d2l.try_gpu()
    net = d2l.RNNModel(rnn_layer, vocab_size=len(vocab))
    net = net.to(device)
    d2l.train_ch8(net, train_iter, vocab, lr, num_epochs, device)


if __name__ == '__main__':
    train()
