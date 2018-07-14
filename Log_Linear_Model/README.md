# Log linear model
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
        log_linear_model.py: 初始版本的代码
        log_linear_model_v2.py: 使用特征提取优化后的代码
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
        'iterator': 100,                          #最大迭代次数
        'stop_iterator': 10,                      #迭代stop_iterator次性能没有提升则结束
        'batch_size': 50,                         #batch_size
        'regularization': False,                  #是否正则化
        'step_opt': False,                        #是否步长优化（模拟退火）
        'C': 0.01,                                #正则化系数
        'eta': 1.0,                               #初始步长
        'decay_steps': 5000                      #衰减速度,数据量越大，值越大,小数据5000,大数据50000
    }
    
    $ cd ./src
    $ python3 log_linear-model.py                   #执行初始版本
    $ python3 log_linear-model_v2.py                #执行特征提取优化版本
### 3.参考结果
#### (1)小数据测试

```
训练集：data/train.conll
开发集：data/dev.conll
```

| 部分特征优化 |            步长优化            |   正则化    |  迭代次数  | train准确率 | dev准确率 | 时间/迭代  |
| :----: | :------------------------: | :------: | :----: | :------: | :----: | :----: |
|   ×    |             ×              |    ×     | 99/100 | 100.00%  | 87.21% | 1.5min |
|   ×    |             ×              | C=0.0001 | 31/42  |  99.99%  | 87.43% | 1.5min |
|   ×    | eta=0.5,  decay_steps=5000 |    ×     | 23/34  | 100.00%  | 87.42% | 1.5min |
|   ×    | eta=0.5,  decay_steps=5000 | C=0.0001 | 27/38  | 100.00%  | 87.52% | 1.5min |
|   √    |             ×              |    ×     | 49/60  | 100.00%  | 87.44% |   8s   |
|   √    |             ×              | C=0.0001 | 12/23  | 100.00%  | 87.55% |  12s   |
|   √    | eta=0.5,  decay_steps=5000 |    ×     |  8/19  | 100.00%  | 87.58% |  13s   |
|   √    | eta=0.5,  decay_steps=5000 | C=0.0001 | 25/36  | 100.00%  | 87.56% |   8s   |

#### (2)大数据测试

```
训练集：big-data/train.conll
开发集：big-data/dev.conll
测试集：big-data/test.conll
```

| 部分特征优化 |            步长优化             |    正则化     | 迭代次数  | train准确率 | dev准确率 | test准确率 | 时间/迭代 |
| :----: | :-------------------------: | :--------: | :---: | :------: | :----: | :-----: | :---: |
|   ×    |              ×              |     ×      | 22/33 |  98.73%  | 93.05% | 92.71%  | 25min |
|   ×    |              ×              | C=0.000001 | 32/43 |  98.82%  | 93.01% | 92.68%  | 32min |
|   ×    | eta=0.5,  decay_steps=50000 |     ×      | 37/48 |  99.33%  | 93.63% | 93.39%  | 32min |
|   ×    | eta=0.5,  decay_steps=50000 | C=0.000001 | 36/47 |  99.30%  | 93.69% | 93.40%  | 33min |
|   √    |              ×              |     ×      | 24/35 |  99.11%  | 93.25% | 93.00%  | 7min  |
|   √    |              ×              | C=0.000001 | 14/25 |  98.83%  | 93.19% | 92.88%  | 30min |
|   √    | eta=0.5,  decay_steps=50000 |     ×      | 53/64 |  99.35%  | 93.72% | 93.49%  | 5min  |
|   √    | eta=0.5,  decay_steps=50000 | C=0.000001 | 22/33 |  99.27%  | 93.69% | 93.43%  | 21min |

