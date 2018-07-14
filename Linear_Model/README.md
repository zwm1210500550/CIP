# Linear model
## 一、目录文件
    ./data/:
        train.conll: 训练集
        dev.conll: 开发集
    ./big_data/
        train.conll: 训练集
        dev.conll: 开发集
        test.conll: 测试集
    ./result:
        res_usew_v1: 初始版本，小数据测试，使用W作为权重的评价结果
        res_usev_v1: 初始版本，小数据测试，使用V作为权重的评价结果
        res_usew_v2: 使用部分特征优化后，小数据测试，使用W作为权重的结果
        res_usev_v2: 使用部分特征优化后，小数据测试，使用V作为权重的结果
        res_usew_v1_bigdata: 初始版本，大数据测试，使用W作为权重的评价结果
        res_usev_v1_bigdata: 初始版本，大数据测试，使用V作为权重的评价结果
        res_usew_v2_bigdata: 使用部分特征优化后，大数据测试，使用W作为权重的结果
        res_usev_v2_bigdata: 使用部分特征优化后，大数据测试，使用V作为权重的结果
        res: 结果汇总
    ./src:
        linear_model.py: 初始版本的代码
        linear_model_v2.py: 使用特征提取优化后的代码
        config.py: 配置文件，用字典存储每个参数
    ./README.md: 使用说明

## 二、运行
### 1.运行环境
    python 3
### 2.运行方法
    #配置文件中各个参数
    config = {
        'train_data_file': '../data/train.conll', #训练集文件,大数据改为'../big_data/train.conll'
        'dev_data_file': '../data/dev.conll',     #开发集文件,大数据改为'../big_data/dev.conll'
        'test_data_file': '../data/dev.conll',    #测试集文件,大数据改为'../big_data/test.conll'
        'averaged': False,                        #是否使用averaged percetron
        'iterator': 50,                           #最大迭代次数
        'stop_iterator': 10,                      #迭代stop_iterator次性能没有提升则结束
    }
    
    $ cd ./src
    $ python3 linear-model.py                   #执行初始版本
    $ python3 linear-model_v2.py                #执行特征提取优化版本
### 3.参考结果
#### (1)小数据测试

```
训练集：data/train.conll
开发集：data/dev.conll
```

|   文件   | linear-model.py | linear-model.py | linear-model_v2.py | linear-model_v2.py |
| :----: | :-------------: | :-------------: | :----------------: | :----------------: |
|  特征权重  |        w        |        v        |         w          |         v          |
|  执行时间  |     0:15:17     |     0:17:41     |      0:07:23       |      0:08:43       |
| 最优迭代轮次 |      12/23      |      14/25      |        9/20        |       10/21        |
| 训练集准确度 |     99.98%      |     98.62%      |       99.96%       |       98.87%       |
| 开发集准确度 |     84.93%      |     85.78%      |       85.69%       |       85.67%       |

#### (2)大数据测试

```
训练集：big-data/train.conll
开发集：big-data/dev.conll
测试集：big-data/test.conll
```

|   文件   | linear-model.py | linear-model.py | linear-model_v2.py | linear-model_v2.py |
| :----: | :-------------: | :-------------: | :----------------: | :----------------: |
|  特征权重  |        w        |        v        |         w          |         v          |
|  执行时间  |    11:49:39     |     7:08:34     |      6:43:16       |      3:08:10       |
| 最优迭代轮次 |      30/41      |      12/23      |       32/43        |       17/28        |
| 训练集准确度 |     98.86%      |     98.40%      |       99.01%       |       99.08%       |
| 开发集准确度 |     92.92%      |     93.73%      |       93.11%       |       93.85%       |
| 测试集准确度 |     92.59%      |     93.48%      |       92.76%       |       93.61%       |

