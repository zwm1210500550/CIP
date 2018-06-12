## 对数线性模型Log-Linear-Model

### 一、目录文件

```
./data/:
    train.conll: 训练集
    dev.conll: 测试集
    origin.txt: 初始版本
    origin-opt.txt: 初始版本，使用步长优化和正则化的结果
    partial-feature.txt: 使用部分特征优化
    partial-feature-opt.txt: 使用部分特征优化后，使用步长优化和正则化的结果
./PPT:
    存放相关的PPT内容，来自于http://hlt.suda.edu.cn/~zhli/teach/cip-2015-fall/
./src:
    log-linear-model.py: 初始版本的代码
    log-linear-model-partial-feature.py: 使用partial-feature优化后的代码
./README.md: 使用说明
```



### 二、运行

##### 1.运行环境

​    python 3.6.3

##### 2.运行方法

```bash
cd ./Log-Linear-Model
python src/log-linear-model.py 								#不使用正则化和步长优化
python src/log-linear-model.py optimize                     #使用正则化和步长优化
python src/log-linear-model-partial-feature.py    			#不使用正则化和步长优化
python src/log-linear-model-partial-feature.py  optimize	#使用正则化和步长优化
```

##### 3.参考结果

| 文件         | log-linear-model.py | log-linear-model.py | log-linear-model-partial-feature.py | log-linear-model-partial-feature.py |
| :----------- | ------------ | ------------ | ------------ | ------------ |
| 是否正则化和步长优化 | 否 | 是 | 否 | 是 |
| 训练集准确率 | 100%  | 99.85% | 100% | 99.99% |
| 测试集准确率 | 86.81% | 87.19% | 87.11% | 87.34% |
| 迭代次数     | 20     | 9   | 19     | 8      |
| 最大迭代次数 | 20      | 20    | 20       | 20       |

注：使用特征优化方法后，实际上拓宽了特征空间，所以准确率一般情况下都有所提高。
