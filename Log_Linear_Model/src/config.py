
config = {
    'train_data_file': '../big_data/train.conll',    #训练集文件
    'dev_data_file': '../big_data/dev.conll',        #开发集文件
    'test_data_file': '../big_data/test.conll',       #测试集文件
    'iterator': 100,                             #最大迭代次数
    'stop_iterator': 10,                        #迭代stop_iterator次性能没有提升则结束
    'batch_size': 50,                            #batch_size
    'regularization': True,                    #是否正则化
    'step_opt': False,                           #是否步长优化（模拟退火）
    'C': 0.0001,                                   #正则化系数
    'eta': 1.0,                                 #初始步长
    'decay_steps': 50000                          #衰减速度,数据量越大，值越大
}
