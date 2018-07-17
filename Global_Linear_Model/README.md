# Global Linear model
## 一、目录文件
    ./data/:
        train.conll: 训练集
        dev.conll: 开发集
    ./big_data/
        train.conll: 训练集
        dev.conll: 开发集
        test.conll: 测试集
    ./result:
        res_v1_usew_small_data: 初始版本，小数据测试，使用W作为权重的评价结果
        res_v1_usev_small_data: 初始版本，小数据测试，使用V作为权重的评价结果
        res_v2_usew_small_data: 使用部分特征优化后，小数据测试，使用W作为权重的结果
        res_v2_usev_small_data: 使用部分特征优化后，小数据测试，使用V作为权重的结果
        res_v1_usew_big_data: 初始版本，大数据测试，使用W作为权重的评价结果
        res_v1_usev_big_data: 初始版本，大数据测试，使用V作为权重的评价结果
        res_v2_usew_big_data: 使用部分特征优化后，大数据测试，使用W作为权重的结果
        res_v2_usev_big_data: 使用部分特征优化后，大数据测试，使用V作为权重的结果
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
        'test_data_file': 'None',    			  #测试集文件,大数据改为'../big_data/test.conll'
        'averaged': False,                        #是否使用averaged percetron
        'iterator': 100,                          #最大迭代次数
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

| partial feature | averaged percetron | 迭代次数  | train 准确率 | dev 准确率 | 时间/迭代 |
| :-------------: | :----------------: | :---: | :-------: | :-----: | :---: |
|        ×        |         ×          | 24/35 |  99.99%   | 87.31%  |  45s  |
|        ×        |         √          | 9/20  |  99.62%   | 87.94%  |  45s  |
|        √        |         ×          | 14/25 |  100.00%  | 87.50%  |  10s  |
|        √        |         √          | 14/25 |  99.78%   | 88.16%  |  10s  |

#### (2)大数据测试

```
训练集：big-data/train.conll
开发集：big-data/dev.conll
测试集：big-data/test.conll
```

| partial feature | averaged percetron | 迭代次数  | train 准确率 | dev 准确率 | test 准确率 | 时间/迭代  |
| :-------------: | :----------------: | :---: | :-------: | :-----: | :------: | :----: |
|        ×        |         ×          | 26/37 |  99.03%   | 93.54%  |  93.37%  | 22min  |
|        ×        |         √          | 10/21 |  98.52%   | 94.40%  |  94.19%  | 22min  |
|        √        |         ×          | 22/33 |  99.04%   | 93.68%  |  93.28%  | 4.5min |
|        √        |         √          | 13/24 |  99.10%   | 94.39%  |  94.21%  | 4.5min |

