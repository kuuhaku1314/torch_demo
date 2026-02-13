"""使用机器翻译展示seq2seq的学习，将长度可变的输入的seq通过编码器转为固定长度的中间状态，
再从中间状态通过解码器转为长度可变的输出序列"""
from torch import nn
import d2l
import torch


def train():
    raw_text = d2l.read_data_nmt()
    text = d2l.preprocess_nmt(raw_text)
    source, target = d2l.tokenize_nmt(text)
    d2l.show_list_len_pair_hist(['source', 'target'], '# tokens per sequence',
                                'count', source, target)
    src_vocab = d2l.Vocab(source, min_freq=2,
                          reserved_tokens=['<pad>', '<bos>', '<eos>'])
    d2l.plt.show()
    d2l.truncate_pad(src_vocab[source[0]], 10, src_vocab['<pad>'])

    class Seq2SeqDecoder(d2l.Decoder):
        """用于序列到序列学习的循环神经网络解码器"""

        def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                     dropout=0., **kwargs):
            super(Seq2SeqDecoder, self).__init__(**kwargs)
            self.embedding = nn.Embedding(vocab_size, embed_size)
            self.rnn = nn.GRU(embed_size + num_hiddens, num_hiddens, num_layers,
                              dropout=dropout)
            self.dense = nn.Linear(num_hiddens, vocab_size)

        def init_state(self, enc_outputs, *args):
            """返回编码器输出的隐状态，用来初始化解码器的隐状态"""
            return enc_outputs[1]

        def forward(self, X, state):
            # 输入的X形状(batch_size, num_steps)
            # 输出'X'的形状：(num_steps,batch_size,embed_size)
            X = self.embedding(X).permute(1, 0, 2)
            # state形状为(num_layers,batch_size,num_hiddens),获取最后一个隐状态作为context
            # 广播context，使其具有与X相同的num_steps，这里是在时间步维度上重复num_steps次
            # context形状为(num_steps,batch_size,num_hiddens)
            context = state[-1].repeat(X.shape[0], 1, 1)
            # 与输入的X在embed_size维度上拼接，扩展了词向量，也就是把隐状态作为了词向量的一部分
            X_and_context = torch.cat((X, context), 2)
            output, state = self.rnn(X_and_context, state)
            # 中间表示映射到输出
            output = self.dense(output).permute(1, 0, 2)
            # output的形状:(batch_size,num_steps,vocab_size)
            # state的形状:(num_layers,batch_size,num_hiddens)
            return output, state

    embed_size, num_hiddens, num_layers, dropout = 32, 32, 2, 0.1
    batch_size, num_steps = 64, 10
    lr, num_epochs, device = 0.005, 300, d2l.try_gpu()

    train_iter, src_vocab, tgt_vocab = d2l.load_data_nmt(batch_size, num_steps)
    encoder = d2l.Seq2SeqEncoder(len(src_vocab), embed_size, num_hiddens, num_layers,
                                 dropout)
    decoder = Seq2SeqDecoder(len(tgt_vocab), embed_size, num_hiddens, num_layers,
                             dropout)
    net = d2l.EncoderDecoder(encoder, decoder)
    d2l.train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device)
    engs = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .']
    fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
    for eng, fra in zip(engs, fras):
        translation, attention_weight_seq = d2l.predict_seq2seq(
            net, eng, src_vocab, tgt_vocab, num_steps, device)
        print(f'{eng} => {translation}, bleu {d2l.bleu(translation, fra, k=2):.3f}')


if __name__ == '__main__':
    train()
