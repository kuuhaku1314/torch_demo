"""LSTM类似GRU, 使用遗忘门，输入门，输出门来进行长短期记忆"""
from torch import nn
import d2l
import torch


def train():
    batch_size, num_steps = 32, 35
    train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)

    def get_lstm_params(vocab_size, num_hiddens, device):
        num_inputs = num_outputs = vocab_size

        def normal(shape):
            return torch.randn(size=shape, device=device) * 0.01

        def three():
            return (normal((num_inputs, num_hiddens)),
                    normal((num_hiddens, num_hiddens)),
                    torch.zeros(num_hiddens, device=device))

        W_xi, W_hi, b_i = three()  # 输入门参数
        W_xf, W_hf, b_f = three()  # 遗忘门参数
        W_xo, W_ho, b_o = three()  # 输出门参数
        W_xc, W_hc, b_c = three()  # 候选记忆元参数
        # 输出层参数
        W_hq = normal((num_hiddens, num_outputs))
        b_q = torch.zeros(num_outputs, device=device)
        # 附加梯度
        params = [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc,
                  b_c, W_hq, b_q]
        for param in params:
            param.requires_grad_(True)
        return params

    def init_lstm_state(batch_size, num_hiddens, device):
        return (torch.zeros((batch_size, num_hiddens), device=device),
                torch.zeros((batch_size, num_hiddens), device=device))

    def lstm(inputs, state, params):
        [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc, b_c,
         W_hq, b_q] = params
        (H, C) = state
        outputs = []
        for X in inputs:
            I = torch.sigmoid((X @ W_xi) + (H @ W_hi) + b_i)
            F = torch.sigmoid((X @ W_xf) + (H @ W_hf) + b_f)
            O = torch.sigmoid((X @ W_xo) + (H @ W_ho) + b_o)
            C_tilda = torch.tanh((X @ W_xc) + (H @ W_hc) + b_c)
            C = F * C + I * C_tilda
            H = O * torch.tanh(C)
            Y = (H @ W_hq) + b_q
            outputs.append(Y)
        # 多返回了一个记忆元C，类似于gru里的候选隐状态C2, C_tilda则是H1，关系类似于H2 = RXC2+(1-R)H1
        return torch.cat(outputs, dim=0), (H, C)

    vocab_size, num_hiddens, device = len(vocab), 256, d2l.try_gpu()
    num_epochs, lr = 500, 1
    model = d2l.RNNModelScratch(len(vocab), num_hiddens, device, get_lstm_params,
                                init_lstm_state, lstm)
    d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, device)
    d2l.plt.show()

    # pytorch实现
    num_inputs = vocab_size
    lstm_layer = nn.LSTM(num_inputs, num_hiddens)
    model = d2l.RNNModel(lstm_layer, len(vocab))
    model = model.to(device)
    d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, device)
    d2l.plt.show()


if __name__ == '__main__':
    train()
