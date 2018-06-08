## 一阶隐马尔可夫模型HMM

### 一、目录文件

```
./data/:
    train.conll: 训练集
    dev.conll: 测试集
    predict.txt: 预测结果
./PPT:
    存放相关的PPT内容，来自于http://hlt.suda.edu.cn/~zhli/teach/cip-2015-fall/
./src:
    HMM.py: 一阶隐马尔可夫模型的代码
./README.md: 使用说明
```



### 二、运行

##### 1.运行环境

​    python 3.6.3

##### 2.运行方法

```bash
cd ./HMM
python src/HMM.py
```

##### 3.参考结果

```
正在评估测试集...
共1910个句子
共50319个单词，预测正确38110个单词
准确率：0.757368
共耗时0:00:24.219168
```

注：旧版的数据(包括测试集和训练集)准确率达到89%左右。

