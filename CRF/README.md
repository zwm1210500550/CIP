## 条件随机场Condition-Random-Field

### 一、目录文件

```
./data/:
    train.conll: 训练集
    dev.conll: 开发集
./big-data/
    train.conll: 训练集
    dev.conll: 开发集
    test.conll: 测试集
./result:
    origin.txt: 初始版本，小数据测试
    partial_feature: 使用部分特征优化后，小数据测试
./src:
    CRF.py: 初始版本的代码
    CRF-partial-feature.py: 使用特征优化的代码
    config.py: 配置文件，用字典存储每个参数
./README.md: 使用说明
```



### 二、运行

##### 1.运行环境

​    python 3.6.3

##### 2.运行方法

```python
#配置文件中各个参数
config = {
    'train_data_file': './data/train.conll',   #训练集文件,大数据改为'./big-data/train.conll'
    'dev_data_file': './data/dev.conll',       #开发集文件,大数据改为'./big-data/dev.conll'
    'test_data_file': './data/dev.conll',      #测试集文件,大数据改为'./big-data/test.conll'
    'batchsize': 1,                            #是否使用averaged percetron
    'iterator': 20,                            #最大迭代次数
    'shuffle': False                           #每次迭代是否打乱数据
}
```

```bash
$ cd ./CRF
$ python src/CRF.py                    #修改config.py文件中的参数
$ python src/CRF-partial-feature.py    #修改config.py文件中的参数
```

##### 3.参考结果

##### (1)小数据测试

训练集：data/train.conll

开发集：data/dev.conll

| 文件         | CRF.py | CRF.py | CRF-partial-feature.py | CRF-partial-feature.py |
| :----------- | ------------ | ------------ | --------------- | --------------- |
| 是否正则化 | 否 |  | 否 |  |
| 是否步长优化 | 否 |  | 否 |  |
| 是否打乱数据 | 否 | 否 | 否 | 否 |
| 训练集准确率 | 100% |  | 100% |  |
| 开发集准确率 | 88.03% |  | 88.47% |  |
| 迭代次数     | 19 |  | 20 |  |
| 最大迭代次数 | 20           | 20           | 20 | 20 |

##### (2)大数据测试

训练集：big-data/train.conll

开发集：big-data/dev.conll

测试集：big-data/test.conll

| 文件         | CRF.py | CRF.py | CRF-partial-feature.py | CRF-partial-feature.py |
| :----------- | ------ | ------ | ---------------------- | ---------------------- |
| 是否正则化   | 否     |        | 否                     |                        |
| 是否步长优化 | 否     |        | 否                     |                        |
| 是否打乱数据 | 否     | 否     | 否                     | 否                     |
| 训练集准确率 |        |        |                        |                        |
| 开发集准确率 |        |        |                        |                        |
| 测试集准确率 |        |        |                        |                        |
| 迭代次数     |        |        |                        |                        |
| 最大迭代次数 | 20     | 20     | 20                     | 20                     |

