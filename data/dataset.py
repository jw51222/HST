import os
import os.path as osp

import numpy as np
import scipy.io as scio

import torch
from scipy.sparse import coo_matrix
from torch_geometric.data import Data, InMemoryDataset, DataLoader


class MyDataset(InMemoryDataset):
    def __init__(self, root, userNum=3, transform=None, pre_transform=None):
        # 初始化时，调用父类 InMemoryDataset 的构造函数。
        # root：数据集存放的根目录。
        # userNum：用户数量，默认是 3。
        # transform 和 pre_transform 是可选的，用于数据处理。
        super(MyDataset, self).__init__(root, transform, pre_transform)

        # 设置用户数量 K
        self.K = userNum

        # 加载已经处理好的数据，保存在 processed_paths 中
        self.data, self.slices = torch.load(self.processed_paths[0],weights_only=False)

    @property
    def raw_file_names(self):
        # 返回原始数据的文件名，通常为一个 MATLAB 文件
        return ['data.mat']

    @property
    def processed_file_names(self):
        # 返回处理后数据的文件名，保存为 PyTorch 的数据格式
        return ['data.pt']

    # 处理数据的方法，读取原始数据并生成符合 PyG 数据格式的对象
    def process(self):
        # 在这里，我们手动设置了用户数量 K = 4
        self.K = 4

        # 读取原始的 MATLAB 数据（使用 scipy.io 的 loadmat 函数）
        raw_h = scio.loadmat(self.raw_paths[0])["h_set"]  # 获取 h_set
        print(raw_h.shape)  # 打印 h_set 的形状，用于检查数据

        raw_w = scio.loadmat(self.raw_paths[0])["w_set"]  # 获取 w_set

        # 初始化一个空的列表，用于存储每个数据样本
        data_list = []

        # 遍历 h_set 和 w_set 中的每个样本，进行处理
        for i in range(raw_h.shape[0]):
            # 提取每一个样本的 h 和 w
            h = raw_h[i]
            w = raw_w[i]

            # 构建图的邻接矩阵
            # adj 是一个 KxK 的邻接矩阵，表示图中节点之间的连接情况
            # 如果节点 i 和节点 j 之间有连接，则 adj[i, j] = 1；否则为 0
            adj = np.zeros([self.K, self.K])
            for i in range(self.K):
                for j in range(self.K):
                    if i != j:
                        adj[i, j] = 1  # 这里假设每个节点与其他节点之间都有边

            # 将邻接矩阵转换为稀疏矩阵的 COO 格式
            adj = coo_matrix(adj)

            # 获取稀疏矩阵的行列索引，这些是边的起始节点和目标节点
            indices = np.vstack((adj.row, adj.col))  # row 和 col 分别是边的起始和终止节点
            edge_index = torch.LongTensor(indices)  # 将索引转换为 PyTorch 张量，符合 PyG 格式

            # 构建特征矩阵 x
            # 这里我们将 h 转置并使用 torch.from_numpy 转换为 PyTorch 张量
            # x 的形状为 [userNum, N_t]，即每个用户的特征
            x = torch.from_numpy(np.swapaxes(h, 0, 1))

            # 构建目标矩阵 y，y 的形状也为 [userNum, N_t]
            y = torch.from_numpy(np.swapaxes(w, 0, 1))

            # 创建一个 PyG 数据对象 Data，将边索引、特征和目标存入其中
            data = Data(edge_index=edge_index, x=x, y=y)

            # 将该数据对象加入数据列表
            data_list.append(data)

        # 使用 collate 方法将数据列表打包成一个批次，得到 processed 数据
        data_save, data_slices = self.collate(data_list)

        # 将处理后的数据保存到磁盘中
        torch.save((data_save, data_slices), self.processed_paths[0])


if __name__ == '__main__':
    # 创建 MyDataset 对象，传入数据集路径和用户数量
    train_dataset = MyDataset('../dataset/data_4u_8n/train', userNum=4)

    # 打印数据集的长度
    print(train_dataset.len())

    # 获取数据集中的第一个样本
    data = train_dataset[0]

    # 打印样本的特征（x）、目标（y）以及边索引（edge_index）的形状
    print(data.x.shape)
    print(data.x)
    print(data.y.shape)
    print(data.y)
    print(data.edge_index.shape)
    print(data.edge_index)
