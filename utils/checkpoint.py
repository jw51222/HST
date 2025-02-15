import os           # 导入操作系统模块，用于执行系统相关的操作，常见的使用场景包括文件路径操作和环境变量的管理。
import numpy as np   # 导入 NumPy 包，用于处理数值数据，提供强大的数组操作和优化的数学运算功能。
import torch        # 导入 PyTorch 包，用于构建和训练深度学习模型，支持基于 GPU 的加速和高度抽象的神经网络构建模块。

# Modified from <url id="culab181gem61voo7apg" type="url" status="parsed" title="stargan-v2/core/checkpoint.py at master · clovaai/stargan-v2" wc="741">https://github.com/clovaai/stargan-v2/blob/master/core/checkpoint.py</url>
# 早停和模型保存

class EarlyStopping:  # 定义一个名为 EarlyStopping 的类，用于实现早停机制和模型保存功能。
    """Early stops the training if validation loss doesn't improve after a given patience."""
    # 早停机制的核心思想是：如果验证集的损失在指定的耐心值（patience）内没有改善，则停止训练。

    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt',  # 定义类的初始化方法，接受多个参数。
                 mode='min', save=True, trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.  # 耐心值，表示在验证损失提升后等待多少个 epoch 再次提升。
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.  # 是否打印每次验证损失提升的消息。
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.  # 监控指标的最小变化量，用于判断是否算作提升。
                            Default: 0
            path (str): Path for the checkpoint to be saved to.  # 保存模型检查点的路径。
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.  # 追踪打印的函数，默认为 print。
                            Default: print
            mode : "min、max"  # 模式，可选值为 "min" 或 "max"，表示监控的指标是否需要最小化或最大化。
        """
        self.patience = patience  # 设置耐心值（patience）为传入的参数，默认为 7。
        self.verbose = verbose    # 设置是否打印详细信息（verbose）为传入的参数，默认为 False。
        self.counter = 0          # 初始化计数器（counter）为 0，用于记录验证损失未改善的次数。
        self.best_score = None    # 初始化最佳分数（best_score）为 None，用于记录目前最好的监控指标值。
        self.early_stop = False   # 初始化早停标志（early_stop）为 False，表示未触发早停。
        self.val_loss_min = np.inf  # 初始化验证损失的最小值（val_loss_min）为无穷大。
        self.delta = delta        # 设置最小变化量（delta）为传入的参数，默认为 0。
        self.path = path          # 设置保存模型检查点的路径（path）为传入的参数，默认为 'checkpoint.pt'。
        self.trace_func = trace_func  # 设置追踪打印的函数（trace_func）为传入的参数，默认为 print。
        self.save = save          # 设置是否保存模型（save）为传入的参数，默认为 True。
        self.mode = mode          # 设置监控模式（mode）为传入的参数，可以是 "min" 或 "max"。

    def __call__(self, val_loss, model):  # 定义类的调用方法，允许使用对象来调用。
        # 根据模式计算分数（score）。
        if self.mode == 'min':  # 如果模式为 "min"，表示需要最小化监控指标。
            score = -val_loss   # 以负的验证损失作为分数，以便可以通过比较大小来判断是否提升。
        else:                   # 如果模式为 "max"，表示需要最大化监控指标。
            score = val_loss    # 以验证损失作为分数。

        if self.best_score is None:  # 如果是第一次调用，或没有记录最佳分数。
            self.best_score = score  # 将当前分数记录为最佳分数。
            if self.save:            # 如果需要保存模型。
                self.save_checkpoint(val_loss, model)  # 调用 save_checkpoint 方法保存模型。
        elif score < self.best_score + self.delta:  # 如果当前分数没有比最佳分数提升足够的 delta。
            self.counter += 1  # 增加计数器（counter）的值。
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')  # 打印早停计数器的信息。
            if self.counter >= self.patience:  # 如果计数器超过耐心值（patience）。
                self.early_stop = True  # 触发早停机制。
        else:  # 如果当前分数比最佳分数有足够提升。
            self.best_score = score    # 更新最佳分数为当前分数。
            if self.save:              # 如果需要保存模型。
                self.save_checkpoint(val_loss, model)  # 调用 save_checkpoint 方法保存模型。
            self.counter = 0           # 重置计数器为 0。

    def save_checkpoint(self, val_loss, model):  # 定义保存模型检查点的方法。
        '''Saves model when validation loss decrease.'''
        # 当验证损失下降时，保存模型。
        if self.verbose:  # 如果需要打印详细信息。
            self.trace_func(  # 打印验证损失下降的消息。
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)  # 使用 PyTorch 的 torch.save 方法保存模型的状态字典。
        self.val_loss_min = val_loss  # 更新验证损失的最小值为当前验证损失。