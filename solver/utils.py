import torch
from tqdm import tqdm

# 训练一个epoch的函数
# train_one_epoch 函数用于在一个 epoch 内对深度学习模型进行训练。它接收模型、优化器、数据加载器、当前 epoch 数、配置参数和损失函数作为输入。函数的主要作用包括：
# 初始化损失和准确率：train_loss 和 train_rate 用于累积整个 epoch 的损失和准确率。
# 进度条显示（可选）：如果启用了进度条（通过 args.progress_bar 设置），使用 tqdm 包对数据加载器进行包装，以便在训练过程中实时显示进度。
# 遍历数据批次：从数据加载器中按批次加载数据，每个批次的数据都会被移动到指定的设备（如 GPU）上。
# 前向传播：使用模型对当前批次的数据进行预测，生成输出。
# 计算损失和准确率：调用损失函数计算当前批次的损失和准确率，同时将数据转换为复数类型以匹配模型的输入要求。
# 反向传播和参数更新：清空优化器的梯度，计算当前批次的梯度，并根据梯度更新模型参数。
# 返回平均损失和准确率：返回本次 epoch 的平均损失和平均准确率，通过将累积的损失和准确率除以批次数量得到。
def train_one_epoch(model, optimizer, data_loader, epoch, args, loss):
    # 初始化训练集损失和训练率
    train_loss = 0.0  # 总损失初始化为0
    train_rate = 0.0  # 总准确率初始化为0

    # 如果设置了显示进度条
    if args.progress_bar:
        data_loader = tqdm(data_loader)  # 用tqdm包包装data_loader，使其显示进度条

    # 遍历数据加载器中的每一个batch
    for step, batch in enumerate(data_loader):
        # 将每个batch的数据移动到指定的设备（如GPU）
        batch = batch.to(args.device)

        # 使用模型对当前batch进行预测
        out = model(batch)

        # 计算当前batch的损失和准确率
        # batch.y.to(torch.complex64)和batch.x.to(torch.complex64)将数据转换为复数类型
        batch_loss, batch_rate = loss(out, batch.y.to(torch.complex64), batch.x.to(torch.complex64))

        # 累加当前batch的损失和准确率
        train_loss += batch_loss.item()  # item()用于获取tensor中的数值
        train_rate += batch_rate.item()  # 同上

        # 如果设置了显示进度条，更新进度条的描述
        if args.progress_bar:
            data_loader.desc = "[train epoch {}]".format(epoch)

        # 反向传播计算梯度
        optimizer.zero_grad()  # 在每次更新之前，将之前的梯度清零
        batch_loss.backward()  # 计算当前batch的梯度
        optimizer.step()  # 根据计算出的梯度更新模型参数

    # 返回本次epoch的平均损失和平均准确率
    return train_loss / (step + 1), train_rate / (step + 1)


# # 模型评估函数（不计算梯度）
# evaluate 函数用于在不计算梯度的情况下，对深度学习模型在一个数据集上进行评估。它接收模型、数据加载器、当前 epoch 数、配置参数和损失函数作为输入。函数的主要作用包括：
# 初始化损失和准确率：val_loss 和 val_rate 用于累积整个数据集的损失和准确率。
# 进度条显示（可选）：如果启用了进度条（通过 args.progress_bar 设置），使用 tqdm 包对数据加载器进行包装，以便在评估过程中实时显示进度。
# 遍历数据批次：从数据加载器中按批次加载数据，每个批次的数据都会被移动到指定的设备（如 GPU）上。
# 前向传播：使用模型对当前批次的数据进行预测，生成输出。
# 计算损失和准确率：调用损失函数计算当前批次的损失和准确率，同时将数据转换为复数类型以匹配模型的输入要求。
# 返回平均损失和准确率：返回验证集的平均损失和平均准确率，通过将累积的损失和准确率除以批次数量得到。
# 总结
# evaluate 函数执行了一个完整的数据集评估流程，包括数据加载、前向传播、损失计算和进度条显示，提供了验证集上的平均损失和准确率，方便用户评估模型的性能。
@torch.no_grad()
def evaluate(model, data_loader, epoch, args, loss):
    # 初始化验证集损失和验证集准确率
    val_loss = 0.0  # 总损失初始化为0
    val_rate = 0.0  # 总准确率初始化为0

    # 如果设置了显示进度条
    if args.progress_bar:
        data_loader = tqdm(data_loader)  # 用tqdm包包装data_loader，使其显示进度条

    # 遍历数据加载器中的每一个batch
    for step, batch in enumerate(data_loader):
        # 将每个batch的数据移动到指定的设备（如GPU）
        batch = batch.to(args.device)

        # 使用模型对当前batch进行预测
        out = model(batch)

        # 计算当前batch的损失和准确率
        # batch.y.to(torch.complex64)和batch.x.to(torch.complex64)将数据转换为复数类型
        batch_loss, batch_rate = loss(out, batch.y.to(torch.complex64), batch.x.to(torch.complex64))

        # 累加当前batch的损失和准确率
        val_loss += batch_loss.item()  # item()用于获取tensor中的数值
        val_rate += batch_rate.item()  # 同上

        # 如果设置了显示进度条，更新进度条的描述
        if args.progress_bar:
            data_loader.desc = "[val epoch {}]".format(epoch)

    # 返回验证集上的平均损失和准确率
    return val_loss / (step + 1), val_rate / (step + 1)


# train_one_epoch 和 evaluate 是深度学习中用于训练和评估模型的两个关键函数，其主要区别如下：
# 1. 目的不同
# train_one_epoch：用于训练模型一个 epoch，目标是更新模型的权重，使模型在训练数据上表现更好。
# evaluate：用于评估模型在验证集或测试集上的性能，是为了查看模型的泛化能力。
# 2. 是否计算梯度
# train_one_epoch：在模型训练过程中会计算梯度，并通过反向传播更新模型参数。
# evaluate：评估过程中不计算梯度，使用 @torch.no_grad() 装饰器，以节省内存并加快计算速度。
# 3. 是否更新模型参数
# train_one_epoch：更新模型参数，通过优化器调整权重以最小化损失。
# evaluate：不更新模型参数，仅评估现有模型参数的性能。
# 4. 返回值不同
# train_one_epoch：返回训练集的平均损失和准确率，用于监控训练过程。
# evaluate：返回验证集或测试集的平均损失和准确率，用于评估模型的泛化性能。
# 5. 应用场景
# train_one_epoch：在模型训练阶段，通常在训练循环中多次调用。
# evaluate：在训练过程中定期调用（如每个 epoch 后），或在模型训练完成后评估最终性能。