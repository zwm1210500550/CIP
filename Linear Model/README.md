#Linear model


##一、目录文件
    ./data/:
        train.conll: 训练集
        dev.conll: 开发集
    ./big-data/
        train.conll: 训练集
        dev.conll: 开发集
        test.conll: 测试集
    ./result:
        res_usew_v1: 初始版本，大数据测试，使用W作为权重的评价结果
        res_usev_v1: 初始版本，大数据测试，使用V作为权重的评价结果
        res_usew_v2: 使用部分特征优化后，大数据测试，使用W作为权重的结果
        res_usev_v2: 使用部分特征优化后，大数据测试，使用V作为权重的结果
        res: 结果汇总
    ./src:
        linear_model.py: 初始版本的代码
        linear_model_v2.py: 使用特征提取优化后的代码
        config.py: 配置文件，用字典存储每个参数
    ./README.md: 使用说明

##二、运行
###1.运行环境
    python 3
###2.运行方法
    #配置文件中各个参数
    config = {
        'train_data_file': './big-data/train.conll',   #训练集文件
        'dev_data_file': './big-data/dev.conll',       #开发集文件
        'test_data_file': './big-data/test.conll',      #测试集文件
        'averaged': False,                         #是否使用权重累加
        'iterator': 20,                            #最大迭代次数
    }
    
    $ cd ./Linear Model
    $ python3 src/linear-model.py                   #执行初始版本
    $ python3 src/linear-model_v2.py                #执行特征提取优化版本
###3.参考结果
####(1)小数据测试

```
训练集：data/train.conll
开发集：data/dev.conll
```

|   文件   | linear-model.py | linear-model.py | linear-model_v2.py | linear-model_v2.py |
| :----: | :-------------: | :-------------: | :----------------: | :----------------: |
|  特征权重  |        w        |        v        |         w          |         v          |
|  执行时间  |     0:10:52     |     0:09:18     |      0:07:09       |      0:06:16       |
| 最优迭代轮次 |      8/20       |      12/20      |       10/20        |       10/20        |
| 训练集准确度 |     99.78%      |     98.84%      |      100.00%       |       98.81%       |
| 开发集准确度 |     84.53%      |     85.42%      |       85.41%       |       85.44%       |

#### (2)大数据测试

```
训练集：big-data/train.conll
开发集：big-data/dev.conll
测试集：big-data/test.conll
```

|   文件   | linear-model.py | linear-model.py | linear-model_v2.py | linear-model_v2.py |
| :----: | :-------------: | :-------------: | :----------------: | :----------------: |
|  特征权重  |        w        |        v        |         w          |         v          |
|  执行时间  |     5:39:14     |     5:40:46     |      3:02:12       |      3:08:10       |
| 最优迭代轮次 |      20/20      |      16/20      |       12/20        |       14/20        |
| 训练集准确度 |     98.53%      |     98.68%      |       98.55%       |       98.94%       |
| 开发集准确度 |     92.77%      |     93.77%      |       92.88%       |       93.83%       |
| 测试集准确度 |     92.33%      |     93.50%      |       92.60%       |       93.57%       |
