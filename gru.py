"""GRU使用重置门R和更新门Z来控制隐状态的更新。
重置门R控制上一个隐状态H对候选隐状态H_tilda的影响：H_tilda = tanh(W_x*X + W_h*(R*H) + b)
更新门Z控制新旧隐状态的混合比例：H_new = Z * H_old + (1-Z) * H_tilda
当Z接近1时保留旧信息，接近0时采用新信息"""
from torch import nn
import d2l
import torch


def train():
    batch_size, num_steps = 32, 35
    train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)

    def get_params(vocab_size, num_hiddens, device):
        num_inputs = num_outputs = vocab_size

        def normal(shape):
            return torch.randn(size=shape, device=device) * 0.01

        def three():
            return (normal((num_inputs, num_hiddens)),
                    normal((num_hiddens, num_hiddens)),
                    torch.zeros(num_hiddens, device=device))

        W_xz, W_hz, b_z = three()  # 更新门参数
        W_xr, W_hr, b_r = three()  # 重置门参数
        W_xh, W_hh, b_h = three()  # 候选隐状态参数
        # 输出层参数
        W_hq = normal((num_hiddens, num_outputs))
        b_q = torch.zeros(num_outputs, device=device)
        # 附加梯度
        params = [W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q]
        for param in params:
            param.requires_grad_(True)
        return params

    def init_gru_state(batch_size, num_hiddens, device):
        return (torch.zeros((batch_size, num_hiddens), device=device),)

    def gru(inputs, state, params):
        W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q = params
        H, = state
        outputs = []
        for X in inputs:
            Z = torch.sigmoid((X @ W_xz) + (H @ W_hz) + b_z)
            R = torch.sigmoid((X @ W_xr) + (H @ W_hr) + b_r)
            H_tilda = torch.tanh((X @ W_xh) + ((R * H) @ W_hh) + b_h)
            H = Z * H + (1 - Z) * H_tilda
            Y = H @ W_hq + b_q
            outputs.append(Y)
        return torch.cat(outputs, dim=0), (H,)

    vocab_size, num_hiddens, device = len(vocab), 256, d2l.try_gpu()
    num_epochs, lr = 500, 1
    model = d2l.RNNModelScratch(len(vocab), num_hiddens, device, get_params,
                                init_gru_state, gru)
    d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, device)
    d2l.plt.show()

    # pytorch实现
    num_inputs = vocab_size
    gru_layer = nn.GRU(num_inputs, num_hiddens)
    model = d2l.RNNModel(gru_layer, len(vocab))
    model = model.to(device)
    d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, device)
    d2l.plt.show()


if __name__ == '__main__':
    train()
