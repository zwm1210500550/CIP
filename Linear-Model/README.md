## 线性模型Linear-Model

### 一、目录文件

```
./data/:
    train.conll: 训练集
    dev.conll: 测试集
    origin_W.txt: 初始版本，使用W作为权重的评价结果
    origin_V.txt: 初始版本，使用V作为权重的评价结果
    partial_feature_W: 使用部分特征优化后，使用W作为权重的结果
    partial_feature_V: 使用部分特征优化后，使用V作为权重的结果
./PPT:
    存放相关的PPT内容，来自于http://hlt.suda.edu.cn/~zhli/teach/cip-2015-fall/
./src:
    Linear_Model.py: 初始版本的代码
    Linear_Model.V2.py: 优化后的代码
./README.md: 使用说明
```



### 二、运行

##### 1.运行环境

​    python 3.6.3

##### 2.运行方法

```bash
cd ./Linear-Model
python src/Linear_Model.py                 #使用W作为权重
python src/Linear_Model.py averaged        #使用V作为权重
python src/Linear_Model_V2.py              #使用W作为权重
python src/Linear_Model_V2.py averaged     #使用V作为权重
```

##### 3.参考结果

| 文件         | Linear_Model | Linear_Model | Linear_Model_V2 | Linear_Model_V2 |
| :----------- | ------------ | ------------ | --------------- | --------------- |
| 特征权重     | W            | V            | W               | V               |
| 执行时间     | 503s         | 522s         | 94s             | 97s             |
| 训练集准确率 | 99.97%       | 99.74%       | 99.99%          | 98.41%          |
| 测试集准确率 | 84.49%       | 85.50%       | 85.65%          | 85.55%          |
| 迭代次数     | 18           | 10           | 9               | 10              |
| 最大迭代次数 | 20           | 20           | 20              | 20              |

分析：使用特征优化方法后，实际上拓宽了特征空间，所以准确率都有所提高。用V作为权重评估测试集结果比较稳定，但不一定比W好。有待改正。[样例代码请点击此处](https://github.com/KiroSummer/LinearModel)。源代码用python2.7编写，改为python3后V比W低，原因未知。
