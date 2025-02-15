from torch_geometric.data import DataLoader
from data.dataset import MyDataset


def get_train_loader(train_path, user_num, batch_size, num_workers=4, **kwargs):
    """
    获取训练数据的 DataLoader。

    参数：
    - train_path: 训练集数据存储路径。
    - user_num: 用户数量，用于构建数据集。
    - batch_size: 每个批次的样本数量。
    - num_workers: 加载数据时使用的线程数，默认是 4。
    - **kwargs: 其他可选参数，可以用于定制数据加载行为。

    返回：
    - 返回一个 DataLoader 对象，用于批量加载训练数据。
    """
    # 加载训练数据集
    dataset = MyDataset(train_path, userNum=user_num)

    # 返回一个 DataLoader 实例，提供批量数据加载功能
    return DataLoader(dataset=dataset,
                      batch_size=batch_size,  # 每批次加载样本数
                      shuffle=True,  # 在训练时启用数据洗牌
                      num_workers=num_workers)  # 使用指定数量的工作线程来加载数据


def get_val_loader(val_path, user_num, batch_size, num_workers=4, **kwargs):
    """
    获取验证数据的 DataLoader。

    参数：
    - val_path: 验证集数据存储路径。
    - user_num: 用户数量，用于构建数据集。
    - batch_size: 每个批次的样本数量。
    - num_workers: 加载数据时使用的线程数，默认是 4。
    - **kwargs: 其他可选参数，用于定制数据加载行为。

    返回：
    - 返回一个 DataLoader 对象，用于批量加载验证数据。
    """
    # 加载验证数据集
    dataset = MyDataset(val_path, userNum=user_num)

    # 返回一个 DataLoader 实例，提供批量数据加载功能
    return DataLoader(dataset=dataset,
                      batch_size=batch_size,  # 每批次加载样本数
                      shuffle=False,  # 在验证阶段不需要洗牌，保持数据顺序
                      num_workers=num_workers)  # 使用指定数量的工作线程来加载数据


def get_test_loader(test_path, user_num, batch_size, num_workers=4, **kwargs):
    """
    获取测试数据的 DataLoader。

    参数：
    - test_path: 测试集数据存储路径。
    - user_num: 用户数量，用于构建数据集。
    - batch_size: 每个批次的样本数量。
    - num_workers: 加载数据时使用的线程数，默认是 4。
    - **kwargs: 其他可选参数，用于定制数据加载行为。

    返回：
    - 返回一个 DataLoader 对象，用于批量加载测试数据。
    """
    # 加载测试数据集
    dataset = MyDataset(test_path, userNum=user_num)

    # 返回一个 DataLoader 实例，提供批量数据加载功能
    return DataLoader(dataset=dataset,
                      batch_size=batch_size,  # 每批次加载样本数
                      shuffle=False,  # 在测试阶段不需要洗牌，保持数据顺序
                      num_workers=num_workers)  # 使用指定数量的工作线程来加载数据
