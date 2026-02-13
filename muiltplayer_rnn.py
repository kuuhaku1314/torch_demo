"""多层LSTM，通过堆叠多个LSTM层来增加模型的表达能力，每层LSTM使用遗忘门、输入门、输出门来进行长短期记忆"""
from torch import nn
import d2l


def train():
    batch_size, num_steps = 32, 35
    train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)

    vocab_size, num_hiddens, num_layers = len(vocab), 256, 2
    num_inputs = vocab_size
    device = d2l.try_gpu()
    lstm_layer = nn.LSTM(num_inputs, num_hiddens, num_layers)
    model = d2l.RNNModel(lstm_layer, len(vocab))
    model = model.to(device)
    num_epochs, lr = 500, 2
    d2l.train_ch8(model, train_iter, vocab, lr * 1.0, num_epochs, device)
    d2l.plt.show()


if __name__ == '__main__':
    train()
