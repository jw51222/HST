import torch
import torch.nn as nn

# 《EE_Unsuper模型功能通俗概述》
# EE_Unsuper 是一个用于评估波束成形性能的模型，主要功能是计算波束向量在给定信道条件下的能效，并通过比较预测波束向量和真实波束向量下的能效来计算损失和相对误差。以下是对其功能的通俗解释：
# 1. 输入数据处理
# 模型接收三个输入：raw_w_hat（波束向量的估计值）、raw_w（真实的波束向量）和 raw_h（信道矩阵）。这些输入数据会被调整维度以便后续计算。
# 2. 数据分组与循环处理
# 将输入数据分成若干组，每组包含 self.K 个用户。对于每组数据，模型会分别处理信道矩阵、真实波束向量和估计波束向量。
# 3. 能效计算
# 对于每组数据，模型通过 get_ee 方法计算能效。能效的计算基于以下步骤：
# 计算信号强度 I，表示波束向量与信道矩阵的相互作用。
# 计算发射功率 P，表示每个用户的发射功率。
# 计算干扰加噪声功率 dr_temp，表示每个用户受到的干扰和噪声。
# 计算速率 R，表示每个用户的通信速率。
# 最终，能效 EE 是速率之和与发射总功率及相关功率的比值。
# 4. 损失计算
# 模型通过以下方式计算损失：
# loss：基于预测波束向量下的能效与期望目标的差异，通常希望最大化能效。
# loss_rate：计算预测能效与真实能效之间的相对误差，以评估模型的预测精度。
# 5. 输出结果
# 模型返回平均损失和平均相对误差，用于评估模型的性能。
# 总结
# EE_Unsuper 模型的主要功能是通过比较预测波束向量和真实波束向量下的能效，评估波束成形性能，并计算相应的损失和相对误差，
# 从而优化波束向量的生成，提高通信系统的能效。

# 定义 EE_Unsuper 类，继承自 PyTorch 的 nn.Module
class EE_Unsuper(nn.Module):
    def __init__(self, user_num):
        # 调用父类的构造函数
        super(EE_Unsuper, self).__init__()
        # 初始化用户数量
        self.K = user_num
        # 设置常数功率消耗（P_c）
        self.P_c = 0.1
        # 设置噪声功率
        self.noise = 0.01

    def forward(self, raw_w_hat, raw_w, raw_h):
        # raw_w_hat: 波束向量的估计值
        # raw_w: 真实的波束向量
        # raw_h: 信道矩阵

        # 调整输入数据的维度，使得信道矩阵、波束向量和估计波束向量的维度便于后续计算
        # 交换张量的一维和二维，我的理解就是矩阵转置操作
        h = torch.swapaxes(raw_h, 0, 1)
        w = torch.swapaxes(raw_w, 0, 1)
        w_hat = torch.swapaxes(raw_w_hat, 0, 1)
        # 初始化损失和相对误差
        # 使用 .to("cpu") 确保数据在 CPU 上，避免 GPU 上的潜在错误
        loss_rate = torch.zeros(1).to("cpu")
        loss = torch.zeros(1).to("cpu")

        # 将数据分成若干组，每组包含 self.K 个用户
        # raw_h.shape[0] 表示总用户数，每组包含 self.K 个用户，循环次数为总用户数除以每组用户数
        for i in range(raw_h.shape[0] // self.K):
            # 提取当前组的信道矩阵、真实波束向量和估计波束向量
            h_sample = h[:, i * self.K:(i + 1) * self.K]
            w_hat_sample = w_hat[:, i * self.K:(i + 1) * self.K]
            w_sample = w[:, i * self.K:(i + 1) * self.K]

            # 计算真实波束向量对应的能效
            ee = self.get_ee(h_sample, w_sample)
            # 计算估计波束向量对应的能效
            ee_hat = self.get_ee(h_sample, w_hat_sample)

            # 累加损失，目标是最大化能效（因此使用负号使最小化损失等同于最大化能效）
            loss -= ee_hat
            # 累加相对误差，衡量预测值与真实值之间的差异
            loss_rate += abs(ee - ee_hat) / ee

        # 返回平均损失和平均相对误差
        # 除以循环次数（即组数）得到平均值
        return loss / (raw_h.shape[0] // self.K), loss_rate / (raw_h.shape[0] // self.K)

    def get_ee(self, h, w):
        # 计算能效值（EE）

        # 计算信号强度 I，基于波束向量与信道矩阵的相互作用
        # torch.matmul(w.T.conj(), h) 表示波束向量转置共轭与信道矩阵的乘积
        # torch.multiply(...) 与共轭相乘得到信号强度的模平方
        I = torch.real(torch.multiply(torch.matmul(w.T.conj(), h), torch.matmul(w.T.conj(), h).conj()))

        # 计算每个用户的发射功率 P
        # torch.matmul(w.T.conj(), w) 得到波束向量与自己共轭的乘积矩阵
        # torch.diag(...) 提取对角线元素，表示各用户的发射功率
        P = torch.real(torch.diag(torch.matmul(w.T.conj(), w)))

        # 计算干扰加噪声功率 dr_temp
        # torch.sum(I, dim=0) 计算每行元素之和
        # 减去对角线元素（各用户的信号强度）得到干扰成分
        # 加上噪声功率
        dr_temp = torch.sum(I, dim=0) - torch.diag(I) + self.noise

        # 计算各用户的速率 R，基于 SINR（Signal-to-Interference-plus-Noise Ratio）
        # torch.log2(1 + ...) 计算以 2 为底的对数，得到速率
        R = torch.log2(1 + torch.div(torch.diag(I), dr_temp))

        # 计算能效 EE，即速率之和与总功率（发射功率加常数功率消耗）之比
        ee = (torch.sum(R)) / (torch.sum(P) + self.P_c)

        return ee