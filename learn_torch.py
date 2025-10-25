import torch
from torch.distributions import multinomial
import torchvision
from torch import nn
from torch.utils import data
from torchvision import transforms
from torch.nn import functional as F
import warnings

warnings.filterwarnings('ignore')

# 声明一个tensor，张量的轴按照shape形状，从左到右从0开始递增，以索引访问也是如此，如tensor[0]访问的是第0个轴的索引为1的张量
torch.tensor([1, 2])
# 声明元素都为1的tensor，数据类型是float32
torch.ones((10, 5), dtype=torch.float32)
# 声明元素都为0的tensor
torch.zeros((10, 5))
# 求向量的点积，[1, 2] dot [3, 4] = 1x3+2x4 = 11
torch.dot(torch.tensor([1, 2]), torch.tensor([3, 4]))
# 矩阵和向量的乘积，向量先升维求得结果，再降维，形状如同 2x2 mv 2 -> 2x2 mm 2x1 -> 2x1 -> 2
torch.mv(torch.tensor([[1, 2], [3, 4]]), torch.tensor([5, 6]))
# 矩阵和矩阵的乘积，2x2 mm 2x1 -> 2x1
torch.mm(torch.tensor([[1, 2], [3, 4]]), torch.tensor([[5], [6]]))
# 批量执行n个矩阵和n个矩阵的乘积，1x2x2 mm 1x2x1 -> 1x2x1
torch.bmm(torch.tensor([[[1, 2], [3, 4]]]), torch.tensor([[[5], [6]]]))
# 根据参数类型相当于dot，mv，mm，另外存在一种vm情况，如 2 vm 2x2 -> 1x2 mm 2x2 -> 1x2 -> 2
torch.matmul(torch.tensor([5, 6]), torch.tensor([[1, 2], [3, 4]]))

t = torch.tensor([1, 2, 3, 4, 5])
# 每个元素求累计和，[1, 2, 3, 4, 5] -> [1, 3, 6, 10, 15]，对矩阵来说dim 0是逐行叠加(每列从上往下sum)，1是逐列叠加(每行从左往右sum)
t.cumsum(dim=0)
# 求和，[1, 2, 3, 4, 5] -> 15
t.sum()
# 获取元素个数，[1, 2, 3, 4, 5] -> 5
t.numel()
# 变换维度，5 -> 1x5，填入-1的位置会根据其他维度自动算出这个维度的大小
t.reshape([-1, 5])
# 重复，这里形状从5 -> 50
t.repeat(10)
# 返回tensor的numpy数组
t.numpy()
# 返回python标准数字表示，张量只能是标量
t.sum().item()
# 返回tensor的范数
torch.norm(torch.ones((4, 9)))
# 取对数
torch.log(torch.ones((4, 9)))
# 取平方根
torch.sqrt(torch.ones((4, 9)))
# 在0维上拼接tensor，也就是长度翻倍，如5,5就会变成10
torch.cat([t, t], 0)
# 沿着一个新维度拼接tensor，维度会加1，如5,5就会变成2x5，若是3x3 stack 3x3 dim=1，则会是3X2X3，dim=2，则会是3X3X3
torch.stack([t, t], 0)
# 进行轴交换，由(4, 9, 1) -> (9, 1, 4)
torch.permute(torch.ones((4, 9, 1)), (1, 2, 0))
# 增加一个维度，(4) -> (4, 1)
torch.unsqueeze(torch.ones((4,)), 1)
# 减少一个维度，(4, 9, 1) -> (4, 9)，dim若为负数则是从最后一个维度倒数
torch.squeeze(torch.ones((4, 9, 1)), -1)
# 张量展平为向量，重复n次
torch.repeat_interleave(torch.ones((4, 9, 1)), 10)

# 随机产生一个一个1x2维的tensor，值满足0-1间的正态分布 ，requires_grad要求梯度
x = torch.randn(size=(1, 2), requires_grad=True)
# 由于x的requires_grad=True，y在计算后，会在内部保存计算图信息
y = 2 * x
# 向前进行梯度传播，也就是算出y的requires_grad=True自变量的grad，通过x.grad获取，这里x的梯度就是y对矩阵x的导数=[[2, 2]]
# retain_graph表明反向传播后不丢弃计算图，若不加gradient参数只有当结果是标量才能使用
y.sum().backward(retain_graph=True)
# 清除计算后得到的x的梯度信息
x.grad.zero_()
# 相当于y dot torch.ones_like(y)再backward，也就是进行了矩阵的点积，这里是全为1的矩阵进行点积相当于sum()函数
y.backward(gradient=torch.ones_like(y))
# 会舍弃内部的计算图信息，这个时候的u，就像是初始化声明出的一个新tensor，但是和原tensor底层共用一份内存
u = y.detach()

# 生成多项分布，如这里可以看成有3个格子对应三种颜色的球,有一个无限大的球堆每种颜色的球占三分之一，每次抽样抽取5个放入对应格子
d = multinomial.Multinomial(5, (torch.ones([3])) / 3)
# 从分布里抽样2次，结果如[[4., 0., 1.], [1., 3., 1.]]，入参要是size类型
d.sample(torch.Size((2,)))
# 从均值为0，标准差为1的正态分布进行抽样，这里抽取了10x5=50个样本组成了10x5的矩阵
torch.normal(0, 1, (10, 5))
# 生成100个(0, 1)间的随机数
torch.rand(100)

features = torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]])
labels = torch.tensor([[1], [3], [5], [7]])
# 从tensor创建一个数据集，列表第一个元素是特征向量矩阵，第二个元素是标签矩阵
dataset = data.TensorDataset(*[features, labels])
if __name__ == '__main__':
    # 创建一个数据集的加载器，每次加载两个，打乱顺序，使用2个线程加载
    # 注意在windows上若要使用多进程，因为windows下没有fork，这个方法必须在main函数下运行，不然会抛错(原理待研究)
    dataloader = data.DataLoader(dataset, batch_size=2, shuffle=True, num_workers=2)
    for X, y in dataloader:
        for x in X:
            continue
# 一个转换函数，可以将plt image或narray类型转为tensor
trans = [transforms.ToTensor()]
# 若trans是多个函数组成的pipeline，可以使用这个方法组合成单个函数
trans = transforms.Compose(trans)
# datasets是包含了数据集的模块，root是下载到的文件夹，train决定是加载训练还是测试数据，transform是转换函数，download为true则下载
# 若已经下载不会再次下载
torchvision.datasets.FashionMNIST(
    root="data", train=True, transform=trans, download=True)

# 一个顺序流，相当于一个pipeline，第一层是平展化，第二层是一个输入为784，输出为10的线性层
net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))
# 对pipeline里的每个pipe施加一个函数作用
net.apply(fn=lambda k: None)
net = nn.Sequential(nn.Linear(784, 10))
# 用正态分布初始化权重
net[0].weight.data.normal_(0, 0.01)
# 初始化偏置
net[0].bias.data.fill_(0)
# 另一种初始化权重的方式
nn.init.normal_(net[0].weight, std=0.01)
# xavier是一种权重初始化方式，实现了前向传播和反向传播时中间层方差不变的折中，在高斯分布中公式为权重的方差=2/(input+output)
nn.init.xavier_normal_(net[0].weight)
# xavier是一种权重初始化方式，实现了前向传播和反向传播时中间层方差不变的折中，在均匀分布[-a， a]中公式为a的平方=6/(input+output)
nn.init.xavier_uniform_(net[0].weight)
# 初始化为0
nn.init.zeros_(net[0].weight)
# 初始化为2
nn.init.constant_(net[0].weight, 2)
# 获取权重的l2范数
net[0].weight.norm()
# 返回(name, parameter)迭代器
net.named_parameters()
# 另一种访问参数的办法，访问第0层的bias
_ = net.state_dict()['0.bias'].data.grad
assert len(net) == 1, 'net layer size not equal 1'
# 在网络中添加一层，模块名不能含有.
net.add_module('relu2', nn.Sequential(nn.Flatten(), nn.Linear(784, 10)))
# 访问第2个module的第2个module
net[1][1].weight.norm()

# 小批量随机梯度下降，一种初始化方式，指明了权重w和偏置b，其中w的权重衰退惩罚率为0.1
torch.optim.SGD([
    {'params': net[0].weight, 'weight_decay': 0.1},
    {'params': net[0].bias}], lr=0.1)
# 简化的声明方式把每层的权重和偏置平铺成数组返回, 如[W1, B1, W2, B2]
torch.optim.SGD(net.parameters(), lr=0.1)
# 另一种声明方式
torch.optim.SGD([net[0].weight, net[0].bias], lr=0.1)

# 作用于每一个元素，返回max(i, 0)
torch.relu(torch.tensor([-1, 1]))
# 作用于每一个元素，返回1 / (1 + exp(-i))，输出区间为[0, 1]
torch.sigmoid(torch.tensor([-1, 1]))
# 作用于每一个元素，返回(1 - 2exp(-i)) / (1 + 2exp(-i))，输出区间为[-1, 1]
torch.tanh(torch.tensor([-1, 1]))
# 将张量的值裁剪到[1, inf)区间，小于1的值变为1，大于inf的值变为inf（实际上不会有大于inf的值）
torch.clamp(torch.tensor([-1, 1]), 1, float('inf'))
# relu层
nn.ReLU()
# 平坦层，把input的每个sample元素平铺
nn.Flatten()
# 延后初始化linear层，也就是说不需要指定输入的形状，只需要指定输出形状
nn.LazyLinear(256)
# 函数化的relu
F.relu(torch.tensor(1))
# 卷积层，输入输出通道数都是1，卷积核形状是1x2，不使用偏置，步幅为1，不填充
nn.Conv2d(1, 1, kernel_size=(1, 2), bias=False, stride=(1, 1), padding=0)
# 最大池化层，保留区域内的最大特征，输入仍然保持原先维度
nn.MaxPool2d(3, stride=(1, 1), padding=0)
# 平均池化层，保留区域内特征的均值
nn.AvgPool2d(3, stride=(1, 1), padding=0)
# 自适应平均池化层，给出池化后的输出形状，会根据输入自适应地计算核的大小和每次移动的步长，在这里把每个通道的hxw个像素聚合为1x1
nn.AdaptiveAvgPool2d((1, 1))

# 保存张量
torch.save([torch.arange(4), torch.arange(5)], 'test_data/x-file')
# 加载张量
x, y = torch.load('test_data/x-file')
net = nn.Linear(5, 3)
# 保存模型参数
torch.save(net.state_dict(), 'test_data/mlp.params')
# 加载模型参数，注意模型入参要和保存时的一致
nn.Linear(5, 3).load_state_dict(torch.load('test_data/mlp.params'))
model = nn.Sequential(nn.Flatten(), nn.Linear(2, 1), nn.ReLU())
# 保存整个神经网络架构和参数
torch.save(model, 'test_data/model.pt')
m = torch.load('test_data/model.pt', weights_only=False)

# 获取显卡数量
torch.cuda.device_count()
# 返回第一块显卡
torch.device('cuda:0')
# 将张量从内存转到显存中
tensor_in_video_storage = torch.zeros(1).cuda()
# 将张量从显存转到内存中
tensor_in_video_storage.to(torch.device('cpu'))
