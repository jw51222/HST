import os
import torch

# 导入构建模型的函数
from models.build import build_model
# 导入自定义的损失函数
from solver.loss import EE_Unsuper
# 导入训练和评估的函数
from solver.utils import train_one_epoch, evaluate
# 导入用于早停的工具
from utils.checkpoint import EarlyStopping


# 《Solver类功能通俗概述》——————负责训练
# Solver 类是一个用于管理深度学习模型训练过程的类，它负责以下几个方面的功能：
# 1. 初始化
# Solver 类接收一个 args 对象作为输入，该对象包含了所有训练所需的参数，如设备选择、模型类型、学习率等。
# 根据 args 中的参数，Solver 类实例化模型，并将其移动到指定的设备（如 GPU 或 CPU）上。
# 如果训练模式是 train，还会初始化优化器用于更新模型参数，以及学习率调度器用于动态调整学习率。
# 2. 加载预训练模型
# 如果需要从预训练模型开始训练（例如从之前保存的最佳模型继续训练），Solver 类提供方法加载这些预训练模型。
# 3. 训练过程
# Solver 类控制整个训练流程，包括多个 epoch 的循环训练。
# 在每个 epoch 中，模型使用训练数据进行前向传播，并通过计算损失函数来评估预测结果与真实标签之间的差异。
# 然后，通过反向传播和优化器来更新模型的参数，以最小化损失函数。
# 训练过程中，还会定期在验证集上评估模型的性能，并根据验证集的表现调整学习率。
# 4. 学习率调整
# 当验证集上的性能指标（如准确率）不再改善时，学习率调度器会自动降低学习率，以帮助模型更好地优化。
# 5. 早停机制
# Solver 类包含早停机制，用于防止模型过拟合。
# 如果验证集上的性能指标在一定数量的 Epoch 内没有改善，早停机制会触发，停止训练并保存当前的最佳模型。
# 总结
# Solver 类通过集成模型构建、训练、验证、学习率调整和早停机制等功能，为深度学习模型的训练提供了一站式的解决方案。
# 它能够自动处理训练过程中的各种细节，帮助用户更高效地训练和优化模型

# 定义Solver类，负责训练过程中的大部分逻辑
class Solver:
    def __init__(self, args):
        super().__init__()
        self.args = args  # 将传入的参数保存为实例变量
        self.device = torch.device(args.device)  # 设置设备（如GPU或CPU）
        self.net = build_model(args)  # 使用build_model函数构建模型
        self.net = self.net.to(self.device)  # 将模型移动到指定的设备上

        # 如果训练模式为'train'，则设置训练的相关组件
        if args.mode == 'train':
            # 设置优化器，使用Adam优化器
            self.optimizer = torch.optim.Adam(self.net.parameters(),
                                              lr=args.lr,  # 学习率
                                              betas=(0.9, 0.999),  # Adam优化器的beta参数
                                              eps=1e-08,  # 用于数值稳定性的极小值
                                              weight_decay=args.weight_decay)  # 权重衰减（L2正则化）

            # 设置学习率调度器，当验证集上的评估指标没有改善时，降低学习率
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', factor=args.factor,
                                                                        patience=args.patience,
                                                                        verbose=True)  # 使用“ReduceLROnPlateau”策略

            # 设置早停机制，防止过拟合
            self.early_stopping = EarlyStopping(patience=5, verbose=True, delta=1e-4,
                                                path=os.path.join(args.model_dir, "best_model.pth"),
                                                save=args.save_model)  # 如果验证集表现没有改善，保存最佳模型并提前停止

    # 加载已经训练好的模型
    def load_model(self):
        self.net.load_state_dict(
            torch.load(os.path.join(self.args.model_dir, "best_model.pth"), map_location=self.device))  # 从指定路径加载最佳模型参数

    # 从指定路径加载模型
    def load_model_from_path(self, path):
        self.net.load_state_dict(torch.load(path, map_location=self.device))  # 从自定义路径加载模型参数

    # 训练过程
    def train(self, loaders):
        args = self.args  # 获取训练时的参数
        net = self.net  # 获取模型
        optimizer = self.optimizer  # 获取优化器
        scheduler = self.scheduler  # 获取学习率调度器

        train_loader = loaders.train  # 获取训练集数据加载器
        val_loader = loaders.val  # 获取验证集数据加载器

        # 如果设置了开始的epoch大于0，说明有预训练模型，可以加载预训练模型
        if args.start_epoch > 0:
            self.load_model_from_path(self.args.pre_model_path)  # 加载预训练模型的参数
            print("Loading model successfully")  # 打印加载成功的信息

        print('Start training...')  # 打印开始训练的消息
        # 根据参数初始化损失函数
        loss = EE_Unsuper(user_num=args.user_num)  # EE_Unsuper是自定义的损失函数类，传入用户数作为参数

        # 循环训练，遍历每个epoch
        for epoch in range(args.start_epoch + 1, args.end_epoch + 1):
            # 设置当前epoch，方便在数据集加载器中使用
            train_loader.dataset.epoch_now = epoch
            val_loader.dataset.epoch_now = epoch
            self.net.train()  # 将模型设置为训练模式（启用dropout等）

            # 训练一个epoch
            train_loss, train_rate = train_one_epoch(net, optimizer, train_loader, epoch, args, loss)  # 调用train_one_epoch函数执行训练

            # 切换到评估模式，关闭dropout等不必要的操作
            self.net.eval()
            # 评估模型在验证集上的表现
            val_rate: float
            val_loss, val_rate = evaluate(net, val_loader, epoch, args, loss)  # 调用evaluate函数执行评估

            # 学习率下降策略，根据验证集上的准确率（val_rate）调整学习率
            scheduler.step(metrics=val_rate)  # 根据验证集的准确率更新学习率

            # 早停判断，检查是否满足提前停止的条件
            self.early_stopping(val_rate, self.net)  # 根据验证集表现判断是否早停

            # 打印本轮训练和验证的损失与准确率
            print(
                "epoch {},train_loss: {:.4f},train_rate: {:.4f}, val_loss: {:.4f},val_rate: {:.4f}".format(
                    epoch, train_loss, train_rate, val_loss, val_rate))

            # 如果早停机制触发，提前结束训练
            if self.early_stopping.early_stop:
                break  # 退出训练循环
