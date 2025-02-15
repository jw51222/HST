# 导入必要的模块
import argparse  # 用于解析命令行参数
import json  # 用于处理 JSON 格式的数据
import os  # 用于操作文件和目录
import random  # 用于生成随机数
import sys  # 提供与 Python 解释器和操作系统交互的函数
import platform  # 提供与操作系统平台相关的信息

import numpy as np  # 导入 numpy 库，用于进行数组和矩阵计算
from torch.backends import cudnn  # 导入 cudnn 库，用于加速深度学习计算

import torch  # 导入 PyTorch 库
from munch import Munch  # 导入 Munch 库，用于更方便地处理字典

# 从 utils 文件夹中的文件模块导入相关函数
from utils.file import prepare_dirs  # 用于准备目录结构
from utils.misc import get_datetime, str2bool  # 获取当前时间和字符串转布尔值的工具函数


def load_cfg() -> object:
    """
    加载训练参数（一般用 shell 文件）
    :return: 配置文件
    """
    # 有两种方式加载配置：使用 JSON 文件或者命令行参数
    if len(sys.argv) >= 2 and sys.argv[1].endswith('.json'):
        # 如果命令行参数中第一个参数是 JSON 文件
        with open(sys.argv[1], 'r') as f:
            # 打开并读取该 JSON 文件
            cfg = json.load(f)
            cfg = Munch(cfg)  # 将加载的字典数据转化为 Munch 对象，方便属性访问
            if len(sys.argv) >= 3:
                # 如果有第三个命令行参数，设置实验 ID
                cfg.exp_id = sys.argv[2]
            else:
                # 如果没有第三个参数，则输出警告
                print("Warning: using existing experiment dir.")
            if not cfg.about:
                # 如果配置中没有 'about' 字段，设置其默认值
                cfg.about = f"Copied from: {sys.argv[1]}"
    else:
        # 如果没有提供 JSON 文件，则通过命令行参数解析配置
        cfg = parse_args()
        cfg = Munch(cfg.__dict__)  # 将解析得到的命令行参数转化为 Munch 对象

        # 判断是否是 Windows 系统，Windows 上默认设置 num_workers 为 0
        if (platform.system() == 'Windows'):
            nw = 0
        else:
            # 计算可以使用的最大线程数（取 batch_size、CPU 核心数和 40 中的最小值）
            nw = min([os.cpu_count(), cfg.batch_size if cfg.batch_size > 1 else 0, 40])
        cfg.num_workers = nw  # 设置 num_workers 为计算得到的线程数
    return cfg  # 返回加载的配置


def parse_args():
    """
    转化命令行参数
    :return: 命令行参数对象
    """
    parser = argparse.ArgumentParser()  # 创建 ArgumentParser 对象，用于解析命令行参数

    # 关于本次实验的一些信息
    parser.add_argument('--about', type=str, default="")  # 记录实验信息
    parser.add_argument('--exp_id', type=str, help='Folder name and id for this experiment.')  # 实验 ID
    parser.add_argument('--exp_dir', type=str, default='expr')  # 实验保存的目录

    # 元参数，如训练模式、设备选择等
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])  # 模式：训练或测试
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')  # 设备：cuda 或 cpu
    parser.add_argument('--model', type=str, default='ComGAT', choices=['ComGAT', 'ComMLP'])  # 模型类型

    # 模型相关参数
    parser.add_argument('--user_num', type=int, default=3)  # 用户数
    parser.add_argument('--antenna_num', type=int, default=8)  # 天线数

    # 数据集相关参数
    parser.add_argument('--train_path', type=str, required=True)  # 训练集路径
    parser.add_argument('--val_path', type=str, required=True)  # 验证集路径
    parser.add_argument('--test_path', type=str, required=False)  # 测试集路径

    # DataLoader 相关参数
    parser.add_argument('--batch_size', type=int, default=16)  # 批次大小
    parser.add_argument('--num_workers', type=int, default=60)  # 数据加载线程数

    # 训练相关参数
    parser.add_argument('--start_epoch', type=int, default=0, help='开始的训练 epoch')
    parser.add_argument('--end_epoch', default=100, type=int, help='训练的最大 epoch 数')
    parser.add_argument('--save_model', type=str2bool, default=True, help='训练时是否保存模型')
    parser.add_argument('--progress_bar', type=str2bool, default=True, help='训练时是否显示进度条')

    # 测试相关参数
    parser.add_argument('--pre_model_path', type=str, required=False)  # 预训练模型路径
    parser.add_argument('--model_path', type=str, required=False)  # 模型保存路径
    parser.add_argument('--save_result', type=str2bool, default=False, required=False)  # 是否保存测试结果

    # 优化相关参数
    parser.add_argument('--lr', type=float, default=1e-3, help="学习率")  # 学习率
    parser.add_argument('--weight_decay', type=float, default=0, help="权重衰减系数")  # 权重衰减
    parser.add_argument('--factor', type=float, default=5e-1, help="学习率衰减因子")  # 学习率衰减因子
    parser.add_argument('--patience', type=int, default=1)  # 学习率调整耐心值

    # 其他参数
    parser.add_argument('--seed', type=int, default=0, help='随机数种子')  # 随机种子
    parser.add_argument('--cudnn_benchmark', type=str2bool, default=False)  # cudnn 是否优化
    parser.add_argument('--cudnn_deterministic', type=str2bool, default=True)  # cudnn 是否启用确定性算法
    return parser.parse_args()  # 返回解析结果


def setup_cfg(args):
    """
    配置训练参数
    :param args: 命令行参数
    """
    # 设置 cudnn 的优化选项
    cudnn.benchmark = args.cudnn_benchmark
    cudnn.deterministic = args.cudnn_deterministic
    # 设置随机种子以保证实验的可复现性
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # 设置 TensorFlow 的日志级别，减少无关日志输出
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    # 根据不同模式（训练或测试）设置实验 ID
    if args.mode == 'train':
        if args.exp_id is None:
            args.exp_id = get_datetime()  # 如果没有提供实验 ID，则用当前时间作为 ID
    else:
        if args.exp_id is None:
            # 如果没有提供实验 ID，则提示用户输入
            args.exp_id = input("Please input exp_id: ")
        # 如果指定的实验 ID 不存在，则检查是否有类似的 ID，避免重复
        if not os.path.exists(os.path.join(args.exp_dir, args.exp_id)):
            all_existed_ids = os.listdir(args.exp_dir)
            for existed_id in all_existed_ids:
                if existed_id.startswith(args.exp_id + "-"):
                    args.exp_id = existed_id
                    print(f"Warning: exp_id is reset to {existed_id}.")
                    break

    # 设置 Windows 系统下的 num_workers 为 0，因为 Windows 系统对多线程的支持较差
    if os.name == 'nt' and args.num_workers != 0:
        print("Warning: reset num_workers = 0, because running on a Windows system.")
        args.num_workers = 0

    # 设置模型保存的路径
    args.model_dir = os.path.join(args.exp_dir, args.exp_id, "models")

    # 准备所需的目录
   # prepare_dirs([args.log_dir, args.model_dir, args.eval_dir])

def save_cfg():
    print("Saving")
def print_cfg(cfg):
    """
    打印配置
    :param cfg: 配置对象
    """
    # 格式化输出配置文件内容
    print(json.dumps(cfg, indent=4))
