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
| 执行时间     | 472s         | 498s         | 119s             | 121s             |
| 训练集准确率 | 99.75%       | 99.98%       | 99.96%          | 99.78%          |
| 测试集准确率 | 84.48%       | 85.45%       | 85.46%          | 85.71%          |
| 迭代次数     | 15           | 20           | 9               | 14              |
| 最大迭代次数 | 20           | 20           | 20              | 20              |

注：代码参考了样例代码[Github链接](https://github.com/KiroSummer/LinearModel)。使用特征优化方法后，实际上拓宽了特征空间，所以准确率一般情况下都有所提高。由于Linear-Model的权重都是整数，在计算每个tag的score时很有可能出现许多个tag最大分值一样的情况，这时候取哪个作为预测的标签会影响最终的结果（误差在0.1%-0.2%)。在本例中，把所有的tag按字母序排列，从左到右编号，计算最大值时默认取第一个出现的最大值（np.argmax）。而在样例代码中，python2.7的词典顺序乱序，且取的是最后一个最大值。如果改为python3结果会出现不一样！
