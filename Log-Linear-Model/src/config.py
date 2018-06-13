config = {
    'train_data_file': './data/train.conll',  # 训练集文件
    'dev_data_file': './data/dev.conll',  # 开发集文件
    'test_data_file': './data/dev.conll',  # 测试集文件
    'iterator': 20,  # 最大迭代次数
    'batchsize': 50,  # 批次大小
    'shuffle': False,  # 每次迭代是否打乱数据
    'regulization': True,  # 是否正则化
    'step_opt': True,  # 是否步长优化
    'eta': 0.5,  # 初始步长
    'C': 0.0001  # 正则化系数
}
