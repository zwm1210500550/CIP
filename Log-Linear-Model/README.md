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
    partial.txt: 使用部分特征优化
    partial-opt.txt: 使用部分特征优化后，使用步长优化和正则化的结果
    bigdata-origin.txt: 大数据测试，初始版本的结果
    bigdata-partial.txt: 大数据测试，部分特征优化的结果
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
    'exitor': 10,                             # 连续多少个迭代没有提升就退出
    'regulization': False,                    # 是否正则化
    'step_opt': False,                        # 是否步长优化
    'eta': 0.5,                               # 初始步长,step_eta为False时无效
    'C': 0.0001                               # 正则化系数,regulization为False时无效
}
```

```bash
cd ./Log-Linear-Model
python src/log-linear-model.py					# 修改配置文件参数
python src/log-linear-model-partial-feature.py	 # 修改配置文件参数
```

##### 3.参考结果

##### (1)小数据测试

训练集：data/train.conll

开发集：data/dev.conll

| partial-feature | 步长优化 |  正则化  | 打乱数据 | 迭代次数 | dev准确率 | 时间/迭代 |
| :-------------: | :------: | :------: | :------: | :------: | :-------: | :-------: |
|        ×        |    ×     |    ×     |    √     |  64/74   |  87.22%   |    25s    |
|        ×        | eta=0.5  | C=0.0001 |    √     |          |           |           |
|        √        |    ×     |    ×     |    √     |  63/73   |  87.53%   |    4s     |
|        √        | eta=0.5  | C=0.0001 |    √     |          |           |           |

注：使用部分特征优化方法后，实际上拓宽了特征空间，所以准确率一般情况下都有所提高。可以修改不同的参数来测试结果。

##### (2)大数据测试

训练集：big-data/train.conll

开发集：big-data/dev.conll

测试集：big-data/test.conll

| partial-feature | 步长优化 | 正则化 | 打乱数据 | 迭代次数 | dev准确率 | test准确率 | 时间/迭代 |
| :-------------: | :------: | :----: | :------: | :------: | :-------: | ---------- | :-------: |
|        ×        |    ×     |   ×    |    √     |          |           |            |           |
|        ×        |          |        |    √     |          |           |            |           |
|        √        |    ×     |   ×    |    √     |   9/19   |  93.20%   | 92.84%     |   142s    |
|        √        |          |        |    √     |          |           |            |           |

注：使用部分特征优化方法后，实际上拓宽了特征空间，所以准确率一般情况下都有所提高。可以修改不同的参数来测试结果。结果取开发集上最好的一次来评价测试集。