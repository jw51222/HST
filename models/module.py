import torch
from complexPyTorch.complexLayers import ComplexReLU, NaiveComplexBatchNorm1d

from torch import nn
from models.layers import GATv2Conv, ComplexLinear

# 《ComGAT模型功能通俗概述》
# ComGAT 是一个用于处理图数据的神经网络模型，它的主要功能是对图中的节点特征进行多层的复杂变换和处理，最终输出满足功率约束的信号。
# 以下是对该模型功能的通俗解释：
# 1. 特征提取阶段
# 型号利用两个图注意力层（GATv2Conv）提取图中节点的特征。类似于从一张复杂的网络图中，挑出重要的节点信息。
# 每个图注意力层都有多个“小脑袋”（注意力头），它们会从不同的角度去观察和分析节点之间的关系，从而更全面地理解图的结构。
# 2. 特征变换与归一化
# 经过图注意力层处理后的特征，会被送入几个线性变换层（ComplexLinear）。
# 这些层的作用就像是对特征进行一系列数学操作，改变它们的形状和大小。
# 在变换过程中，还会应用归一化处理（NaiveComplexBatchNorm1d），让特征的数值范围更加稳定，
# 避免某些特征过大或过小对后续计算的影响。
# 3. 输出与功率约束
# 最后，模型会输出经过多层处理后的特征，并对输出信号进行归一化，确保其满足功率约束。
# 这就好比在发送信号时，要控制信号的强度，不能让它过大或过小，以保证信号能够稳定地传输和接收。
# 总结
# ComGAT 模型的主要功能是通过对图数据进行多层的复杂变换和处理，提取出重要的节点特征，并输出满足功率约束的信号。
# 它在图数据处理领域，如社交网络分析、分子结构预测等任务中，能够有效地捕捉图的结构信息，为后续的分析和预测提供有力支持。
class ComGAT(nn.Module):
    def __init__(self, args):
        super(ComGAT, self).__init__()

        # 初始化第一个 GATv2Conv 层，输入特征数为 antenna_num，输出特征数为 64，使用 20 个头。
        # 负斜率为 0，启用残差连接。
        self.gat1 = GATv2Conv(args.antenna_num, 64, 20, negative_slope=0, residual=True)

        # 初始化第二个 GATv2Conv 层，输入特征数为 20 * 64（由于多头注意力），输出特征数为 512，使用 20 个头。
        # 负斜率为 0，启用残差连接。
        self.gat2 = GATv2Conv(20 * 64, 512, 20, negative_slope=0, residual=True)

        # 初始化复杂的 ReLU 激活函数（ComplexReLU）。
        self.relu = ComplexReLU()

        # 初始化第一个 ComplexLinear 层，输入特征数为 512 * 20，输出特征数为 512。
        self.lin1 = ComplexLinear(512 * 20, 512)

        # 初始化第一个批量归一化层（NaiveComplexBatchNorm1d），用于 512 维特征。
        self.bn1 = NaiveComplexBatchNorm1d(512)

        # 初始化第二个 ComplexLinear 层，输入特征数为 512，输出特征数为 512。
        self.lin2 = ComplexLinear(512, 512)

        # 初始化第二个批量归一化层（NaiveComplexBatchNorm1d），用于 512 维特征。
        self.bn2 = NaiveComplexBatchNorm1d(512)

        # 初始化最终的 ComplexLinear 层，输入特征数为 512，输出特征数为 antenna_num。
        self.lin3 = ComplexLinear(512, args.antenna_num)

        # 保存传入的参数（用于访问设备，例如将张量移动到正确的设备上）。
        self.args = args

        # 初始化功率约束，作为 1 的平方根，在设备上进行缩放。
        self.max_output = torch.sqrt(torch.ones(1)).to(args.device)

    def forward(self, data):
        # 从输入数据中提取特征（x）和边索引（edge_index）。
        x, edge_index = data.x, data.edge_index

        # 将输入特征 x 转换为复数类型（complex64）。
        x = x.to(torch.complex64)

        # 将数据通过第一个 GAT 层。
        x = self.gat1(x, edge_index)

        # 应用复杂的 ReLU 激活函数。
        x = self.relu(x)

        # 将数据通过第二个 GAT 层。
        x = self.gat2(x, edge_index)

        # 应用复杂的 ReLU 激活函数。
        x = self.relu(x)

        # 将数据通过第一个 ComplexLinear 层。
        x = self.lin1(x)

        # 应用批量归一化。
        x = self.bn1(x)

        # 应用复杂的 ReLU 激活函数。
        x = self.relu(x)

        # 将数据通过第二个 ComplexLinear 层。
        x = self.lin2(x)

        # 应用批量归一化。
        x = self.bn2(x)

        # 应用复杂的 ReLU 激活函数。
        x = self.relu(x)

        # 将数据通过最终的 ComplexLinear 层。
        x = self.lin3(x)

        # 计算输出的 L2 范数（欧几里得范数），按维度 1 计算。
        x_norm = torch.norm(x, p=2, dim=1).to(self.args.device)

        # 使用最大范数值对输出进行归一化，以应用功率约束。
        out = self.max_output * torch.div(x,
                                          torch.max(x_norm, torch.ones(x_norm.size()).to(self.args.device)).unsqueeze(1))

        return out


# 《ComMLP模型功能通俗概述》
# ComMLP 是一个用于处理复数数据的多层感知机（MLP）类，其主要功能如下：
# 1. 数据输入与初步处理
# 模型接收输入数据 x，这些数据通常是复数形式。你可以把 x 想象成一组复杂的信号，每个信号由实部和虚部组成，
# 就像声音信号既有振幅也有相位。
# 2. 多层线性变换
# ComMLP 通过对输入数据进行多次线性变换，将其转换为不同维度的特征表示。这些线性变换就像是对数据进行了一系列的数学运算，
# 改变了数据的形状和大小，以便提取出更有用的信息。
# 3. 批量归一化
# 在模型中，有多个批量归一化层。这些层的作用类似于对数据进行 “校准”，确保数据的数值范围在合理的范围内，
# 避免某些数据过大或过小对模型训练的影响。
# 4. 激活函数
# 每次线性变换后，都会应用复杂的 ReLU 激活函数。激活函数就像是一个 “开关”，它决定哪些信息被保留，哪些信息被丢弃，
# 从而为模型引入非线性能力。
# 5. 输出结果
# 最终，模型通过多次线性变换和激活函数，将输入数据转换为输出结果。这个输出结果的形状和大小由模型的最后一个线性层决定。
# 总结
# ComMLP 是一个用于处理复数数据的多层感知机模型，它通过对输入数据进行多次线性变换、批量归一化和激活函数处理，
# 提取出数据的特征，并输出最终的结果。这个模型在处理复数数据（如音频信号、图像信号等）时具有较强的灵活性和表达能力
class ComMLP(nn.Module):

    def __init__(self, args):
        super(ComMLP, self).__init__()

        # 初始化第一个 ComplexLinear 层，输入特征数为 antenna_num * user_num，输出特征数为 256。
        self.fc1 = ComplexLinear(args.antenna_num * args.user_num, 256)

        # 初始化第二个 ComplexLinear 层，输出特征数为 512。
        self.fc2 = ComplexLinear(256, 512)

        # 初始化第三个 ComplexLinear 层，输出特征数为 1024。
        self.fc3 = ComplexLinear(512, 1024)

        # 初始化批量归一化层，用于 1024 维特征。
        self.bn3 = NaiveComplexBatchNorm1d(1024)

        # 初始化第四个 ComplexLinear 层，输出特征数为 512。
        self.fc4 = ComplexLinear(1024, 512)

        # 初始化批量归一化层，用于 512 维特征。
        self.bn4 = NaiveComplexBatchNorm1d(512)

        # 初始化第五个 ComplexLinear 层，输出特征数为 256。
        self.fc5 = ComplexLinear(512, 256)

        # 初始化批量归一化层，用于 256 维特征。
        self.bn5 = NaiveComplexBatchNorm1d(256)

        # 初始化最终的 ComplexLinear 层，输出特征数为 antenna_num * user_num。
        self.fc6 = ComplexLinear(256, args.antenna_num * args.user_num)

        # 初始化复杂的 ReLU 激活函数。
        self.relu = ComplexReLU()

    def forward(self, x):
        # 将数据通过第一个 ComplexLinear 层并应用 ComplexReLU 激活函数。
        x = self.fc1(x)
        x = self.relu(x)

        # 将数据通过第二个 ComplexLinear 层并应用 ComplexReLU 激活函数。
        x = self.fc2(x)
        x = self.relu(x)

        # 将数据通过第三个 ComplexLinear 层并应用批量归一化，再应用 ComplexReLU 激活函数。
        x = self.fc3(x)
        x = self.bn3(x)
        x = self.relu(x)

        # 将数据通过第四个 ComplexLinear 层并应用批量归一化，再应用 ComplexReLU 激活函数。
        x = self.fc4(x)
        x = self.bn4(x)
        x = self.relu(x)

        # 将数据通过第五个 ComplexLinear 层并应用批量归一化，再应用 ComplexReLU 激活函数。
        x = self.fc5(x)
        x = self.bn5(x)
        x = self.relu(x)

        # 将数据通过最终的 ComplexLinear 层（输出层）。
        x = self.fc6(x)
        return x

# ComGAT 和 ComMLP 是两个不同的模型类，
# ComGAT 适用于处理图数据并具有更复杂的网络结构，而 ComMLP 是一个简单的多层感知机，适用于普通的复数数据处理任务