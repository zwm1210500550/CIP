## 对数线性模型Log-Linear-Model

### 一、目录文件

```
./data/:
    train.conll: 小数据训练集
    dev.conll: 小数据开发集
./big-data:
    train.conll: 大数据训练集
    dev.conll: 大数据开发集
    test.conll: 大数据测试集
./result:
    origin.txt: 初始版本
    origin-opt.txt: 初始版本，使用步长优化和正则化的结果
    partial-feature.txt: 使用部分特征优化
    partial-feature-opt.txt: 使用部分特征优化后，使用步长优化和正则化的结果
./src:
    config.py: 配置文件
    log-linear-model.py: 初始版本的代码
    log-linear-model-partial-feature.py: 使用partial-feature优化后的代码
./README.md: 使用说明
```



### 二、运行

##### 1.运行环境

​    python 3.6.3

##### 2.运行方法

```
#配置文件中各个参数
config = {
    'train_data_file': './data/train.conll',  # 训练集文件,大数据改为'./big-data/train.conll'
    'dev_data_file': './data/dev.conll',      # 开发集文件,大数据改为'./big-data/dev.conll'
    'test_data_file': './data/dev.conll',     # 测试集文件,大数据改为'./big-data/test.conll'
    'iterator': 20,                           # 最大迭代次数
    'batchsize': 50,                          # 批次大小
    'shuffle': False,                         # 每次迭代是否打乱数据
    'regulization': False,                    # 是否正则化
    'step_opt': False,                        # 是否步长优化
    'eta': 0.5,                               # 初始步长,step_eta为False时无效
    'C': 0.0001                               # 正则化系数,regulization为False时无效
}
```

```bash
cd ./Log-Linear-Model
python src/log-linear-model.py					# 修改配置文件参数
python src/log-linear-model-partial-feature.py	# 修改配置文件参数
```

##### 3.参考结果

##### (1)小数据测试

训练集：data/train.conll

开发集：data/dev.conll

| 文件         | log-linear-model.py | log-linear-model.py | log-linear-model-partial-feature.py | log-linear-model-partial-feature.py |
| :----------- | ------------ | ------------ | ------------ | ------------ |
| 是否正则化 | 否 | 是,系数0.0001 | 否 | 是,系数0.0001 |
| 是否步长优化 | 否 | 是,初始0.5 | 否 | 是,初始0.5 |
| 是否打乱数据 | 否 | 否 | 否 | 否 |
| 批次大小 | 50 | 50 | 50 | 50 |
| 最大迭代次数 | 20 | 20 | 20 | 20 |
| 训练集准确率 | 100%  | 99.85% | 100% | 99.99% |
| 开发集准确率 | 86.81% | 87.19% | 87.11% | 87.34% |
| 迭代次数     | 20     | 9   | 19     | 8      |

注：使用部分特征优化方法后，实际上拓宽了特征空间，所以准确率一般情况下都有所提高。可以修改不同的参数来测试结果。

##### (2)大数据测试

训练集：big-data/train.conll

开发集：big-data/dev.conll

测试集：big-data/test.conll

| 文件         | log-linear-model.py | log-linear-model.py | log-linear-model-partial-feature.py | log-linear-model-partial-feature.py |
| :----------- | ------------------- | ------------------- | ----------------------------------- | ----------------------------------- |
| 是否正则化   | 否                  | 是,系数0.0001       | 否                                  | 是,系数0.0001                       |
| 是否步长优化 | 否                  | 是,初始0.5          | 否                                  | 是,初始0.5                          |
| 是否打乱数据 | 否                  | 否                  | 否                                  | 否                                  |
| 批次大小     | 50                  | 50                  | 50                                  | 50                                  |
| 最大迭代次数 | 20                  | 20                  | 20                                  | 20                                  |
| 训练集准确率 |                     |                     |                                     |                                     |
| 开发集准确率 |                     |                     |                                     |                                     |
| 测试集准确率 |                     |                     |                                     |                                     |
| 迭代次数     |                     |                     |                                     |                                     |

注：使用部分特征优化方法后，实际上拓宽了特征空间，所以准确率一般情况下都有所提高。可以修改不同的参数来测试结果。结果取开发集上最好的一次来评价测试集。