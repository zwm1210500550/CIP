
config = {
    'train_data_file': '../data/train.conll',    #训练集文件
    'dev_data_file': '../data/dev.conll',    #训练集文件
    'test_data_file': 'None',       #测试集文件
    'iterator': 100,                             #最大迭代次数
    'stop_iterator': 10,                        #迭代stop_iterator次性能没有提升则结束
    'batch_size': 1,                            #batch_size
    'regularization': False,                    #是否正则化
    'step_opt': False,                           #是否步长优化（模拟退火）
    'C': 0.0001,                                   #正则化系数
    'eta': 1.0,                                 #初始步长
}
