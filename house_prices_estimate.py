"""房价预估"""
import numpy as np
import d2l
import pandas as pd
import torch
from torch import nn


def get_k_fold_data(k, i, X, y):
    """把数据分成K折，把第i折作为验证集，剩下折拼在一起"""
    assert k > 1
    fold_size = X.shape[0] // k
    X_train, y_train, X_valid, y_valid = None, None, None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        X_part, y_part = X[idx, :], y[idx]
        if j == i:
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = torch.cat([X_train, X_part], 0)
            y_train = torch.cat([y_train, y_part], 0)
    return X_train, y_train, X_valid, y_valid


def train():
    train_data = pd.read_csv(d2l.download('kaggle_house_train'))
    test_data = pd.read_csv(d2l.download('kaggle_house_test'))
    # 去除无用的id列和训练集中的SalePrice标签
    all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))
    # 获取数字类型的列的索引
    numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
    # 对所有数字列进行归一化
    all_features[numeric_features] = all_features[numeric_features].apply(
        lambda x: (x - x.mean()) / (x.std()))
    # 填充数字列Na的值为0
    all_features[numeric_features] = all_features[numeric_features].fillna(0)
    # “Dummy_na=True”会把“na”(缺失值)视为有效的特征值，也就是说会当成一种分类，并为其创建特征列，对Object这些列扩充为独热向量
    all_features = pd.get_dummies(all_features, dummy_na=True)
    # 获取样本数
    n_train = train_data.shape[0]
    # numpy数组转为tensor
    train_features = torch.tensor(all_features[:n_train].values.astype(float), dtype=torch.float32, device=d2l.try_gpu())
    test_features = torch.tensor(all_features[n_train:].values.astype(float), dtype=torch.float32, device=d2l.try_gpu())
    # 获取训练数据标签
    train_labels = torch.tensor(
        train_data.SalePrice.values.reshape(-1, 1), dtype=torch.float32, device=d2l.try_gpu())
    loss = nn.MSELoss()
    # 获取特征数量
    in_features = train_features.shape[1]

    def get_net():
        net = nn.Sequential(nn.Linear(in_features, 1)).to(device=d2l.try_gpu())
        return net

    def log_rmse(net, features, labels):
        # 为了在取对数时进一步稳定该值，将小于1的值设置为1
        clipped_preds = torch.clamp(net(features), 1, float('inf'))
        rmse = torch.sqrt(loss(torch.log(clipped_preds),
                               torch.log(labels)))
        return rmse.item()

    def _train(net, train_features, train_labels, test_features, test_labels,
               num_epochs, learning_rate, weight_decay, batch_size):
        train_ls, test_ls = [], []
        train_iter = d2l.load_array((train_features, train_labels), batch_size)
        # 这里使⽤的是AdamW优化算法，Adam+weight_decay并不有效
        optimizer = torch.optim.AdamW(net.parameters(),
                                     lr=learning_rate,
                                     weight_decay=weight_decay)
        for epoch in range(num_epochs):
            for X, y in train_iter:
                optimizer.zero_grad()
                l = loss(net(X), y)
                l.backward()
                optimizer.step()
            train_ls.append(log_rmse(net, train_features, train_labels))
            if test_labels is not None:
                test_ls.append(log_rmse(net, test_features, test_labels))
        return train_ls, test_ls

    def k_fold(k, X_train, y_train, num_epochs, learning_rate, weight_decay,
               batch_size):
        train_l_sum, valid_l_sum = 0, 0
        for i in range(k):
            data = get_k_fold_data(k, i, X_train, y_train)
            net = get_net()
            train_ls, valid_ls = _train(net, *data, num_epochs, learning_rate,
                                        weight_decay, batch_size)
            # 取每轮最后一次训练误差
            train_l_sum += train_ls[-1]
            valid_l_sum += valid_ls[-1]
            if i == 0:
                d2l.plot(list(range(1, num_epochs + 1)), [train_ls, valid_ls],
                         xlabel='epoch', ylabel='rmse', xlim=[1, num_epochs],
                         legend=['train', 'valid'], yscale='log')
            print(f'折{i + 1}，训练log rmse{float(train_ls[-1]):f}, '
                  f'验证log rmse{float(valid_ls[-1]):f}')
        return train_l_sum / k, valid_l_sum / k

    k, num_epochs, lr, weight_decay, batch_size = 5, 400, 5, 0, 64
    train_l, valid_l = k_fold(k, train_features, train_labels, num_epochs, lr,
                              weight_decay, batch_size)
    print(f'{k}-折验证: 平均训练log rmse: {float(train_l):f}, '
          f'平均验证log rmse: {float(valid_l):f}')
    d2l.plt.show()

    def train_and_pred(train_features, test_features, train_labels, test_data,
                       num_epochs, lr, weight_decay, batch_size):
        net = get_net()

        train_ls, _ = _train(net, train_features, train_labels, None, None,
                             num_epochs, lr, weight_decay, batch_size)
        d2l.plot(np.arange(1, num_epochs + 1), [train_ls], xlabel='epoch',
                 ylabel='log rmse', xlim=[1, num_epochs], yscale='log')
        print(f'训练log rmse：{float(train_ls[-1]):f}')
        # 将网络应⽤于测试集。
        preds = net(test_features).detach().cpu().numpy()
        # 将其重新格式化以导出到Kaggle
        test_data['SalePrice'] = pd.Series(preds.reshape(1, -1)[0])
        submission = pd.concat([test_data['Id'], test_data['SalePrice']], axis=1)
        submission.to_csv('test_data/submission.csv', index=False)

    train_and_pred(train_features, test_features, train_labels, test_data,
                   num_epochs, lr, weight_decay, batch_size)


if __name__ == '__main__':
    train()
