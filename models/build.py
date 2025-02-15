# 从 GAT_EE.models.module 导入 ComGAT 和 ComMLP 类
from models.module import ComGAT, ComMLP

# 定义一个字典，将模型名称映射到相应的模型类
model_dict = {'ComGAT': ComGAT, 'ComMLP': ComMLP}


# 定义构建模型的函数
def build_model(args):
    # 从传入的args中获取指定的模型名称，并通过字典获取对应的模型类
    model_class = model_dict[args.model]

    # 使用args参数实例化模型
    model = model_class(args)

    # 返回创建的模型实例
    return model
