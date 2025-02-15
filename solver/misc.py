import torch


def get_I(h, w):
    """
    计算 I 矩阵 (信号功率矩阵)
    :param h: 信道值, shape: [antennas, user_num]
    :param w: 波束矩阵, shape: [antennas, user_num]
    :return: I, shape: [K, K]
    """
    K = h.shape[1]  # 取用户数量 K (第二维长度)
    I = torch.zeros([K, K])  # 初始化一个 K x K 的零矩阵 I
    for i in range(K):  # 外层循环，遍历用户 i
        for j in range(K):  # 内层循环，遍历用户 j
            # 计算 w[:, i]^H * h[:, j]
            # w[:, i].T.conj() 表示 w 第 i 列向量的共轭转置
            # h[:, j] 表示 h 第 j 列向量
            temp = torch.matmul(w[:, i].T.conj(), h[:, j])
            # temp * temp.conj() 得到该复数的功率 (取实部)
            I[i, j] = torch.real(temp * temp.conj())
    return I


def get_P(w):
    """
    计算每个用户对应的功率 (波束向量模平方)
    :param w: 波束矩阵, shape: [antennas, user_num]
    :return: w2, shape: [K], 表示每个用户的发射功率
    """
    K = w.shape[1]  # 取用户数量 K
    w2 = torch.zeros(K)  # 初始化长度为 K 的零向量 w2
    for i in range(K):  # 循环遍历每个用户 i
        # 计算 w[:, i]^H * w[:, i] (即波束向量的复数内积)
        w2[i] = torch.real(torch.matmul(w[:, i].T.conj(), w[:, i]))
    return w2


def get_R(noise, I):
    """
    计算每个用户的速率 R (log2(1+SINR))
    :param noise: 噪声功率 (标量)
    :param I: 信号功率矩阵, shape: [K, K], I[i, j]表示第i用户的波束对第j用户的贡献
    :return: R, shape: [K], 每个用户的速率
    """
    K = I.shape[1]  # 用户数量 K
    R = torch.zeros(K)  # 初始化长度为 K 的零向量 R
    dr_temp = torch.zeros(K)  # 存放干扰 + 噪声的临时变量 dr_temp

    for i in range(K):  # 计算干扰及噪声
        dr_temp[i] = noise  # 初始化为噪声功率
        for j in range(K):  # 将其余用户对第 i 用户的干扰加到 dr_temp[i] 中
            if i != j:  # 不包含自身信号
                dr_temp[i] = dr_temp[i] + I[j, i]

    for i in range(K):  # 计算每个用户的速率 R[i] = log2(1 + 信号/干扰)
        tem = torch.div(I[i, i], dr_temp[i])  # 信号 / (干扰 + 噪声)
        R[i] = torch.log2(1 + tem)
    return R


def get_EE(h, w):
    """
    计算能效 (EE): sum(R)/ (sum(P) + P_c)
    :param h: 信道值, shape: [antennas, user_num]
    :param w: 波束矩阵, shape: [antennas, user_num]
    :return: ee (标量), 能效值
    """
    P_c = 0.1  # 常数功率消耗(电路功率等)
    noise = 0.01  # 噪声功率

    # 1) 计算信号功率矩阵 I
    I = get_I(h, w)
    # 2) 计算每个用户的发射功率向量 P
    P = get_P(w)
    # 3) 计算各用户速率 R
    R = get_R(noise, I)
    # 4) 计算能效 EE = sum(R) / (sum(P) + P_c)
    ee = (torch.sum(R)) / (torch.sum(P) + P_c)

    # if (ee == 0):
    #     print("异常")
    #     print(h)
    #     print(w)
    return ee


def get_EEv2(h, w):
    """
    计算能效 (EE) 的向量化版本，与 get_EE 的原理相同，但使用矩阵运算实现。
    :param h: 信道值, shape: [antennas, user_num]
    :param w: 波束矩阵, shape: [antennas, user_num]
    :return: ee (标量), 能效值
    """
    P_c = 0.1  # 常数功率消耗(电路功率等)
    noise = 0.01  # 噪声功率

    # 1) 计算复数乘积, I = (w^H * h) * (w^H * h)^H
    #   torch.matmul(w.T.conj(), h) -> shape: [user_num, user_num]
    #   再与其自身共轭相乘 -> I (K x K), 每个元素 I[i, j] 对应第 i 波束对第 j 用户贡献
    I = torch.real(torch.multiply(
        torch.matmul(w.T.conj(), h),
        torch.matmul(w.T.conj(), h).conj()
    ))

    # 2) 计算每个用户的发射功率(取对角线)
    P = torch.real(torch.diag(torch.matmul(w.T.conj(), w)))

    # 3) 干扰 + 噪声: (每列信号之和) - 对角线(自身信号) + 噪声
    dr_temp = torch.sum(I, dim=0) - torch.diag(I) + noise

    # 4) 计算速率 R[i] = log2(1 + I[i,i] / 干扰)
    R = torch.log2(1 + torch.div(torch.diag(I), dr_temp))

    # 5) EE = sum(R) / (sum(P) + P_c)
    ee = (torch.sum(R)) / (torch.sum(P) + P_c)

    return ee
