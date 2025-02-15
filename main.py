# 从 munch 模块导入 Munch 类，它是一个类似字典的对象，用于将属性访问方式与字典数据结构相结合
from munch import Munch

# 从 config 模块导入相关函数：配置文件操作函数 setup_cfg、load_cfg、save_cfg 和打印配置函数 print_cfg
# from config import setup_cfg, load_cfg, save_cfg, print_cfg
from config import setup_cfg, load_cfg,  print_cfg
# 从 data.loader 模块导入数据加载器函数：获取训练集加载器、测试集加载器和验证集加载器
from data.loader import get_train_loader, get_test_loader, get_val_loader

# 从 solver.solver 模块导入 Solver 类，Solver 类包含训练与测试的具体方法
from solver.solver import Solver


# main 函数是程序的主入口，接收命令行参数或配置文件作为参数
def main(args: object) -> object:
    # 初始化 Solver 类，solver 对象用于管理训练过程
    solver = Solver(args)

    # 根据 args.mode 判断当前模式是训练（'train'）还是测试（'test'）
    if args.mode == 'train':
        # 如果是训练模式，创建训练集和验证集的数据加载器
        loaders = Munch(train=get_train_loader(**args), val=get_val_loader(**args))
        # 调用 solver.train() 方法进行模型训练，传入数据加载器
        solver.train(loaders)
    elif args.mode == 'test':
        # 如果是测试模式，这里没有实现具体逻辑，预留位置用于后续添加测试功能
        pass
    else:
        # 如果模式既不是训练也不是测试，抛出异常并提示未实现的模式
        assert False, f"Unimplemented mode: {args.mode}"


# 该语句块确保当前脚本作为主程序执行时运行
if __name__ == '__main__':
    # 加载配置文件，通过 load_cfg 函数获取配置对象
    cfg = load_cfg()

    # 设置配置，通过 setup_cfg 函数进行一些初始化的配置
    setup_cfg(cfg)

    # 打印当前配置，确保配置内容正确
    print_cfg(cfg)

    # 调用 main 函数，传入配置对象（cfg），开始程序的主逻辑
    main(cfg)

