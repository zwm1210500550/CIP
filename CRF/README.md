# CRF(Condition-Random-Field)
## 一、目录文件
    ./data/:
        train.conll: 训练集
        dev.conll: 开发集
    ./big_data/
        train.conll: 训练集
        dev.conll: 开发集
        test.conll: 测试集
    ./result:
        res_v1_1_smalldata: 初始版本，小数据测试
        res_v1_2_smalldata: 初始版本，使用正则化,小数据测试
        res_v1_3_smalldata: 初始版本，使用模拟退火,小数据测试
        res_v1_4_smalldata: 初始版本，使用正则化和模拟退火,小数据测试
        res_v2_1_smalldata: 特征优化版本，小数据测试
        res_v2_2_smalldata: 特征优化版本，使用正则化,小数据测试
        res_v2_3_smalldata: 特征优化版本，使用模拟退火,小数据测试
        res_v2_4_smalldata: 特征优化版本，使用正则化和模拟退火,小数据测试
        res_v1_1_bigdata: 初始版本，大数据测试
        res_v1_2_bigdata: 初始版本，使用正则化,大数据测试
        res_v1_3_bigdata: 初始版本，使用模拟退火,大数据测试
        res_v1_4_bigdata: 初始版本，使用正则化和模拟退火,大数据测试
        res_v2_1_bigdata: 特征优化版本，大数据测试
        res_v2_2_bigdata: 特征优化版本，使用正则化,大数据测试
        res_v2_3_bigdata: 特征优化版本，使用模拟退火,大数据测试
        res_v2_4_bigdata: 特征优化版本，使用正则化和模拟退火,大数据测试
    ./src:
        CRF.py: 初始版本的代码
        CRF_v2.py: 使用特征提取优化后的代码
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
        'test_data_file': 'None',                 #测试集文件,大数据改为'../big_data/test.conll'
        'iterator': 100,                          #最大迭代次数
        'stop_iterator': 10,                      #迭代stop_iterator次性能没有提升则结束
        'batch_size': 1,                          #batch_size
        'regularization': False,                  #是否正则化
        'step_opt': False,                        #是否步长优化（模拟退火）
        'C': 0.0001,                              #正则化系数
        'eta': 1.0,                               #初始步长   
    }
    
    $ cd ./src
    $ python3 CRF.py                   #执行初始版本
    $ python3 CRF_v2.py                #执行特征提取优化版本
### 3.参考结果
#### (1)小数据测试

```
训练集：data/train.conll
开发集：data/dev.conll
```

| 部分特征优化 |  步长优化   |   正则化    |  迭代次数  | train准确率 | dev准确率 | 时间/迭代 |
| :----: | :-----: | :------: | :----: | :------: | :----: | :---: |
|   ×    |    ×    |    ×     | 99/100 | 100.00%  | 88.38% |  80s  |
|   ×    |    ×    | C=0.0001 | 13/24  |  99.99%  | 88.40% |  80s  |
|   ×    | eta=0.5 |    ×     | 33/44  | 100.00%  | 88.76% |  80s  |
|   ×    | eta=0.5 | C=0.0001 | 22/33  | 100.00%  | 88.64% |  87s  |
|   √    |    ×    |    ×     | 53/64  | 100.00%  | 88.77% |  14s  |
|   √    |    ×    | C=0.0001 | 16/27  | 100.00%  | 88.81% |  16s  |
|   √    | eta=0.5 |    ×     |  7/18  | 100.00%  | 88.84% |  13s  |
|   √    | eta=0.5 | C=0.0001 | 11/22  | 100.00%  | 88.88% |  15s  |

#### (2)大数据测试

```
训练集：big-data/train.conll
开发集：big-data/dev.conll
测试集：big-data/test.conll
```

| 部分特征优化 |  步长优化   |    正则化     | 迭代次数  | train准确率 | dev准确率 | test准确率 | 时间/迭代 |
| :----: | :-----: | :--------: | :---: | :------: | :----: | :-----: | :---: |
|   ×    |    ×    |     ×      | 19/30 |  98.86%  | 93.51% | 93.14%  | 90min |
|   ×    |    ×    | C=0.000001 | 10/21 |  97.84%  | 93.04% | 92.94%  | 75min |
|   ×    | eta=0.5 |     ×      | 30/41 |  99.47%  | 94.09% | 93.81%  | 75min |
|   ×    | eta=0.5 | C=0.000001 | 62/73 |  99.44%  | 94.27% | 94.09%  | 75min |
|   √    |    ×    |     ×      | 14/25 |  99.03%  | 93.70% | 93.38%  | 11min |
|   √    |    ×    | C=0.000001 | 15/26 |  98.57%  | 93.27% | 93.12%  | 45min |
|   √    | eta=0.5 |     ×      | 32/43 |  99.55%  | 94.28% | 93.92%  | 11min |
|   √    | eta=0.5 | C=0.000001 | 60/71 |  99.56%  | 94.38% | 94.16%  | 45min |

