# 导入必要的库
import numpy as np
import torch
from numpy.random import RandomState
from torch.nn import Linear, Parameter, Module
from torch.nn.init import _calculate_correct_fan
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, remove_self_loops, softmax
import torch.nn.functional as F
from torch_geometric.nn.inits import glorot

# 定义一个函数，将实部和虚部分别应用给输入张量，并返回复数类型结果
def apply_complex(fr, fi, input, dtype=torch.complex64):
    # fr 和 fi 分别是对实部和虚部的线性变换 (或更一般的函数)，
    # 其中 input 是一个复数张量
    # 这里用 (a + b i) 表示一个复数，将其拆分为实部 a 和虚部 b 分别传入 fr 和 fi
    # 同时由于复数乘法有 (a + b i)(c + d i) = (ac - bd) + i(ad + bc)，
    # 这里只是把输入的实部、虚部分别送入函数，然后按特定组合方式形成新的复数。
    # 该函数返回值 = [fr(实部) - fi(虚部)] + i [ fr(虚部) + fi(实部) ]
    return (fr(input.real) - fi(input.imag)).type(dtype) \
           + 1j * (fr(input.imag) + fi(input.real)).type(dtype)

# 定义复数形式的 Leaky ReLU 激活函数
def complex_leaky_relu(input, negative_slope):
    # 分别对 input 的实部与虚部应用 F.leaky_relu，再组合成复数
    # 之所以写成 real(...) + 1j * imag(...)，是为了保留复数结构
    return F.leaky_relu(input.real, negative_slope).type(torch.complex64) \
           + 1j * F.leaky_relu(input.imag, negative_slope).type(torch.complex64)


# 复数线性层
#类的功能概括：
# 复数线性变换：实现复数线性层，将复数输入通过两个独立的线性变换（分别针对实部和虚部）后，重新组合成复数输出。
# 自定义初始化：采用基于 Kaiming 的复数权重初始化方法，通过生成模和相位，分别赋值给实部和虚部的权重，确保模型初始化的合理性和稳定性
class ComplexLinear(Module):
    def __init__(self, in_features, out_features, bias=True):
        """
        初始化函数，定义了两个线性层，分别用于处理复数的实部和虚部。
        in_features : 输入特征数
        out_features: 输出特征数
        bias        : 是否带偏置
        """
        super(ComplexLinear, self).__init__()   # 调用父类的构造函数
        self.fc_r = Linear(in_features, out_features, bias=bias)  # 用于处理实部的线性层
        self.fc_i = Linear(in_features, out_features, bias=bias)  # 用于处理虚部的线性层
        self.reset_parameters()

    def forward(self, input):
        """
        前向传播：对一个复数输入 input（形状 [batch_size, in_features] ），
        分别应用 fc_r 和 fc_i，然后通过 apply_complex 合成为复数输出。
        """
        return apply_complex(self.fc_r, self.fc_i, input)

    def reset_parameters(self):
        # 初始化线性层的参数，这里自定义了一个 complex_kaiming_normal_ 方法
        self.complex_kaiming_normal_()

    def complex_kaiming_normal_(self, mode="fan_in"):
        """
        自定义的复数 Kaiming 初始化。
        为 fc_r.weight 和 fc_i.weight 生成模和相位，然后分别赋值给实部和虚部权重。
        """

        # 1. 计算权重初始化时所需的 fan 值。该值取决于权重的形状和 mode 参数。
        #    `mode` 可以是 'fan_in' 或 'fan_out'，影响初始化的大小。
        #    _calculate_correct_fan 是一个帮助函数，返回用于初始化的 fan 数值。
        fan = _calculate_correct_fan(self.fc_r.weight, mode)

        # 2. 根据 fan 值计算 Rayleigh 分布的尺度参数 s。Rayleigh 分布用于生成模（权重的幅值）。
        #    Kaiming 初始化的一个特征是将权重初始化为较小的值，以保持网络的稳定性。
        s = 1. / fan  # Rayleigh 分布的尺度参数

        # 3. 创建一个随机数生成器（RandomState），用于生成随机数。
        #    随后用它来生成模和相位。
        rng = RandomState()

        # 4. 生成一个 Rayleigh 分布的模（即权重的幅度）。
        #    `scale` 参数决定了 Rayleigh 分布的尺度，`size` 是生成的模的形状，和 fc_r.weight 一样。
        #    这一步生成的是模（幅度），即权重的大小。
        modulus = rng.rayleigh(scale=s, size=self.fc_r.weight.shape)

        # 5. 在 -π 到 π 之间均匀随机生成相位。
        #    相位将用于控制权重的方向（实部和虚部）。
        #    `size` 参数决定生成的相位数目，这里和模的形状一致。
        phase = rng.uniform(low=-np.pi, high=np.pi, size=self.fc_r.weight.shape)

        # 6. 使用模和相位生成实部和虚部的权重。
        #    实部权重是模与相位的余弦值的乘积，虚部权重是模与相位的正弦值的乘积。
        #    这将生成复数权重的实部和虚部，符合复数初始化的需求。
        weight_real = modulus * np.cos(phase)  # 实部权重
        weight_imag = modulus * np.sin(phase)  # 虚部权重

        # 7. 将生成的实部和虚部权重转换为 PyTorch 的张量，并赋值给 `fc_r` 和 `fc_i` 的权重。
        #    使用 `torch.nn.Parameter` 将这些权重设置为可训练参数。
        #    dtype=torch.float32 表示使用 32 位浮点数类型存储权重。
        self.fc_r.weight = torch.nn.Parameter(torch.tensor(weight_real, dtype=torch.float32))  # 为实部赋值
        self.fc_i.weight = torch.nn.Parameter(torch.tensor(weight_imag, dtype=torch.float32))  # 为虚部赋值

# GATv2 变种的图注意力层（Graph Attention Layer）
# GATv2Conv 是基于图注意力机制（GAT）的卷积层，用于图数据（图神经网络）中的特征学习与传播。
# 它通过多头注意力机制自适应地为图中不同邻居节点分配权重，从而捕捉节点间特征的复杂依赖关系。
# 该类支持复数特征、残差连接、共享权重等特性，并自动为图结构添加自环以确保每个节点能考虑自身特征。
# 它还提供了基于 Kaiming 的复数权重初始化方法以加速模型收敛，支持多种配置选项以灵活适应不同的任务需求。
# 在前向传播中，它通过线性变换和消息传递实现节点特征的更新，最后根据配置选择拼接或平均多头输出，并可选择性地返回注意力权重。
# 该类适用于图分类、节点分类和链接预测等多种任务。

# GATv2Conv 是一种用于处理图数据的 “神奇的盒子”。这个盒子能把图中点和线之间的复杂关系整理清楚。
# 你可以把图想象成一张有很多点和线的网，比如社交网络中的人和朋友关系，或者化学物质中的原子和键。
# 工作时，这个盒子会用一种特殊的注意力机制，就像一个人在观察一幅画时，会更仔细地看画中有趣的某些部分一样。
# 这里的 “注意力” 会自动决定图中的每个点应该更关注哪些邻近的点，从而更好地理解每个点的特征。
# 这个盒子还能处理复数特征，这就好比看东西时用两种不同的方式去观察，可以捕捉到更多的信息。
# 而且，它会自动调整自己的权重，就像学习新技能时，不断调整方法一样。
# 此外，它还能够添加一种类似于导航提醒的东西，帮助模型更好地处理信息，就像开车时的导航提醒你什么时候该转弯一样。
# 总的来说，GATv2Conv 就是一个强大而智能的工具，能够从复杂的图数据中提取出有价值的信息，帮助我们更好地理解和应用这些数据。
class GATv2Conv(MessagePassing):
    def __init__(
            self,
            in_channels,
            out_channels,
            heads: int = 1,
            concat: bool = True,
            negative_slope: float = 0.2,
            dropout: float = 0.0,
            residual: bool = False,
            add_self_loops: bool = True,
            bias: bool = True,
            share_weights: bool = False,
            **kwargs,
    ):
        """
        初始化函数，配置各种超参数和必要的层。
        in_channels   : 输入特征维度
        out_channels  : 输出特征维度
        heads         : 注意力头的数量
        concat        : 多个头的输出是否进行拼接 (True) 或求平均 (False)
        negative_slope: Leaky ReLU 的负斜率
        dropout       : dropout 概率
        residual      : 是否添加残差连接
        add_self_loops: 是否给图加自环
        bias          : 是否添加偏置
        share_weights : 是否在边的两端共享权重
        kwargs        : 其他 MessagePassing 中需要的关键字参数
        """
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels     # 输入特征维度
        self.out_channels = out_channels   # 输出特征维度
        self.heads = heads                 # 多头注意力的头数
        self.concat = concat               # 多头输出是否拼接
        self.negative_slope = negative_slope  # LeakyReLU 的负斜率
        self.dropout = dropout             # Dropout 概率
        self.add_self_loops = add_self_loops  # 是否加自环
        self.share_weights = share_weights    # 是否在左右两侧共享线性权重
        self.residual = residual           # 是否使用残差连接

        # 定义线性变换：将输入映射到 heads * out_channels 的维度
        self.lin_l = ComplexLinear(in_channels, heads * out_channels, bias=False)  # 左侧顶点特征
        if share_weights:
            # 如果共享权重，则右侧和左侧用同一个复杂线性层
            self.lin_r = self.lin_l
        else:
            # 否则定义独立的右侧线性层
            self.lin_r = ComplexLinear(in_channels, heads * out_channels, bias=False)

        # 注意力参数 att，形状为 [1, heads, out_channels]
        # 在 GATv2 中，注意力向量会在后续算子中与特征相乘
        self.att = Parameter(torch.Tensor(1, heads, out_channels))

        # 根据是否 concat 和是否需要 bias，定义偏置向量
        if bias and concat:
            self.bias = torch.nn.Parameter(torch.zeros(heads * out_channels, dtype=torch.complex64))
        elif bias and not concat:
            self.bias = torch.nn.Parameter(torch.zeros(heads * out_channels, dtype=torch.complex64))
        else:
            self.register_parameter('bias', None)

        # 如果使用残差连接，需要将输入特征映射到输出维度一致后再相加
        if residual:
            if self.in_channels != out_channels * heads:
                # 如果输入维度与输出维度不匹配，就用额外的线性层变换到一致再加
                self.res_fc = ComplexLinear(in_channels, heads * out_channels)
            else:
                # 否则就可以直接把输入加到输出上，不需要额外的映射
                self.res_fc = None
        else:
            self.res_fc = None

        self._alpha = None  # 存储注意力权重（可选，用于可视化或调试）

        # 调用初始化函数
        self.reset_parameters()

    def reset_parameters(self):
        """
        重置参数：这里示例用了 Glorot 初始化 (xavier)，也可以用自己的 complex_kaiming_normal_。
        同时对 att 也进行复数权重的初始化。
        """
        # lin_l, lin_r 的实部和虚部各自进行 glorot 初始化
        glorot(self.lin_l.fc_r.weight)
        glorot(self.lin_l.fc_i.weight)
        glorot(self.lin_r.fc_r.weight)
        glorot(self.lin_r.fc_i.weight)

        # 如果使用残差连接且需要额外映射，也对它的权重进行初始化
        if self.residual and self.res_fc is not None:
            glorot(self.res_fc.fc_r.weight)
            glorot(self.res_fc.fc_i.weight)

        # 初始化注意力向量 att 的实部和虚部
        weight_real_att = torch.Tensor(1, self.heads, self.out_channels)
        weight_imag_att = torch.Tensor(1, self.heads, self.out_channels)
        glorot(weight_real_att)
        glorot(weight_imag_att)
        # 合成为复数并赋值给 self.att
        self.att.data = torch.nn.Parameter(
            torch.tensor(weight_real_att + 1j * weight_imag_att, dtype=torch.complex64)
        )

    def complex_kaiming_normal_(self, mode="fan_in"):
        """
        使用 Kaiming 正态分布来初始化复数权重的示例。
        如果你想要使用此方法，可以在 reset_parameters() 中调用它。
        这里与上面的 complex_kaiming_normal_ 类似，但同时对 lin_l / lin_r / att 做初始化。
        """
        rng = RandomState()

        # 1) 初始化 lin_l
        fan_l = _calculate_correct_fan(self.lin_l.fc_r.weight, mode)
        s_l = 1. / fan_l
        modulus_l = rng.rayleigh(scale=s_l, size=self.lin_l.fc_r.weight.shape)
        phase_l = rng.uniform(low=-np.pi, high=np.pi, size=self.lin_l.fc_r.weight.shape)
        weight_real_l = modulus_l * np.cos(phase_l)
        weight_imag_l = modulus_l * np.sin(phase_l)
        self.lin_l.fc_r.weight = torch.nn.Parameter(torch.tensor(weight_real_l, dtype=torch.float32))
        self.lin_l.fc_i.weight = torch.nn.Parameter(torch.tensor(weight_imag_l, dtype=torch.float32))

        # 2) 初始化 lin_r
        fan_r = _calculate_correct_fan(self.lin_r.fc_r.weight, mode)
        s_r = 1. / fan_r
        modulus_r = rng.rayleigh(scale=s_r, size=self.lin_r.fc_r.weight.shape)
        phase_r = rng.uniform(low=-np.pi, high=np.pi, size=self.lin_r.fc_r.weight.shape)
        weight_real_r = modulus_r * np.cos(phase_r)
        weight_imag_r = modulus_r * np.sin(phase_r)
        self.lin_r.fc_r.weight = torch.nn.Parameter(torch.tensor(weight_real_r, dtype=torch.float32))
        self.lin_r.fc_i.weight = torch.nn.Parameter(torch.tensor(weight_imag_r, dtype=torch.float32))

        # 3) 初始化注意力向量 att
        s_att = 1. / self.heads
        modulus_att = rng.rayleigh(scale=s_att, size=[self.heads, self.out_channels])
        phase_att = rng.uniform(low=-np.pi, high=np.pi, size=[self.heads, self.out_channels])
        weight_real_att = modulus_att * np.cos(phase_att)
        weight_imag_att = modulus_att * np.sin(phase_att)
        self.att.data = torch.nn.Parameter(
            torch.tensor(weight_real_att + 1j * weight_imag_att, dtype=torch.complex64)
        )

    def forward(self, x, edge_index, return_attention_weights: bool = None):
        """
        前向传播：计算图卷积（图注意力）输出。
        x          : 节点特征矩阵，形状 [num_nodes, in_channels]
        edge_index : 边索引，形状 [2, num_edges]
        return_attention_weights: 是否返回注意力权重
        """
        H, C = self.heads, self.out_channels      # 多头数 H，每头输出维度 C
        assert x.dim() == 2                      # 保证输入是二维 (N, in_channels)

        # 1) 先对节点特征做线性变换：左侧
        x_l = self.lin_l(x).view(-1, H, C)       # [N, heads, out_channels]

        # 2) 如果 share_weights, 那么右侧与左侧共享同一个变换
        if self.share_weights:
            x_r = x_l
        else:
            x_r = self.lin_r(x).view(-1, H, C)   # [N, heads, out_channels]

        # 3) 如果需要添加自环，则先移除已有自环，再添加自环
        if self.add_self_loops:
            num_nodes = x.size(0)
            edge_index, _ = remove_self_loops(edge_index)
            edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)

        # 4) 调用 propagate 函数进行消息传递和聚合
        #    x=(x_l, x_r) 表示在消息传递时需要用到的特征，
        #    具体会在 message(), aggregate(), update() 等方法中使用
        out = self.propagate(edge_index, x=(x_l, x_r), size=None)

        # 5) 如果使用了残差连接，则将输入加到输出
        if self.res_fc is not None:
            # 如果有额外的线性变换（输入维度与输出维度不同）
            out = out + self.res_fc(x).view(-1, H, C)
        elif self.residual:
            # 否则直接加上原始输入 (需注意 shapes 是否匹配)
            out = out + x.view(-1, 1, self.in_channels)

        # 6) 如果 concat = True，则拼接多个头，否则对多个头做平均
        if self.concat:
            # [N, H, C] -> [N, H*C]
            out = out.view(-1, H * C)
        else:
            # 平均一下多个头的输出
            out = out.mean(dim=1)

        # 7) 最后加上偏置（如果有）
        if self.bias is not None:
            out = out + self.bias

        # 如果需要返回注意力权重，可在这里返回 (out, self._alpha)
        # self._alpha 通常在 message() 或者 update() 中保存
        if return_attention_weights:
            return out, self._alpha
        else:
            return out
