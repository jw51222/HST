import scipy.io as scio
import os


def process_raw_data(base_path, save_path='./'):
    """
    处理原始数据并将其划分为训练集、验证集和测试集。

    参数：
    - base_path: 原始数据集所在的基本路径。原始数据应当是 `.mat` 格式。
    - save_path: 处理后的数据集保存路径，默认是当前路径。

    返回：
    - 无返回值，但会将处理后的数据保存到指定的文件夹中。
    """

    # 加载原始数据
    # 使用 scipy 的 loadmat 方法读取 .mat 文件，返回一个字典
    # 读取原始数据集 '4u_8n.mat' 中的 "H_gather" 和 "W_gather" 数据
    raw_h = scio.loadmat(os.path.join(base_path, "4u_8n.mat"))["H_gather"]
    raw_w = scio.loadmat(os.path.join(base_path, "4u_8n.mat"))["W_gather"]

    # 输出原始数据的形状，便于检查数据是否正确加载
    print(raw_w.shape)  # 输出 W_gather 的形状
    print(raw_h.shape)  # 输出 H_gather 的形状

    # 数据划分比例，这里假设训练集包含 8 个样本，验证集 1 个样本，测试集 1 个样本
    train_number = 8  # 训练集的样本数
    val_number = 1  # 验证集的样本数
    test_number = 1  # 测试集的样本数

    # 根据指定的比例划分原始数据
    # 将 raw_h 数据集划分为训练集、验证集和测试集
    h_set_train = raw_h[:train_number]  # 训练集
    h_set_val = raw_h[train_number:train_number + val_number]  # 验证集
    h_set_tes = raw_h[train_number + val_number:]  # 测试集

    # 同样地，将 raw_w 数据集划分为训练集、验证集和测试集
    w_set_train = raw_w[:train_number]  # 训练集
    w_set_val = raw_w[train_number:train_number + val_number]  # 验证集
    w_set_tes = raw_w[train_number + val_number:]  # 测试集

    # 打印数据划分的数量，确认划分的正确性
    print("训练集{}，验证集{},测试集{}".format(train_number, val_number, test_number))

    # 将划分好的数据存储为 .mat 格式
    # 保存训练集数据到 'train/raw/data.mat' 路径下
    scio.savemat(os.path.join(os.getcwd(), save_path, "train/raw", "data.mat"),
                 {'h_set': h_set_train, 'w_set': w_set_train})  # 保存训练集

    # 保存验证集数据到 'val/raw/data.mat' 路径下
    scio.savemat(os.path.join(os.getcwd(), save_path, "val/raw", "data.mat"),
                 {'h_set': h_set_val, 'w_set': w_set_val})  # 保存验证集

    # 保存测试集数据到 'test/raw/data.mat' 路径下
    scio.savemat(os.path.join(os.getcwd(), save_path, "test/raw", "data.mat"),
                 {'h_set': h_set_tes, 'w_set': w_set_tes})  # 保存测试集


if __name__ == '__main__':
    # 设置原始数据集的路径和处理后数据集的保存路径
    base_path = '../dataset/raw_data'
    save_path = "../dataset/data_4u_8n"

    # 调用处理函数
    process_raw_data(base_path, save_path)
