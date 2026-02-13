"""双向循环神经网络，利用过去和未来的信息预测中间缺失的信息，一般只使用在填充缺失的单词，机器翻译上，
这里的隐状态翻倍，举个例子，一个序列abcd, 正向隐状态已经学习了ab 推测第三个字符, 然后计算和序列abc损失的时候
反向隐状态已经学习了d, 推测接下来的字符, 计算与序列dc的损失，即理论上正反序列都能学习到, 正反推测序列加起来是整个序列长度"""
from torch import nn
import d2l


def train():
    # 加载数据
    batch_size, num_steps, device = 32, 35, d2l.try_gpu()
    train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)
    # 通过设置“bidirectional=True”来定义双向LSTM模型
    vocab_size, num_hiddens, num_layers = len(vocab), 256, 2
    num_inputs = vocab_size
    lstm_layer = nn.LSTM(num_inputs, num_hiddens, num_layers, bidirectional=True)
    model = d2l.RNNModel(lstm_layer, len(vocab))
    model = model.to(device)
    # 训练模型
    num_epochs, lr = 500, 1
    d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, device)
    d2l.plt.show()


if __name__ == '__main__':
    train()
