# LinearModel

## 运行

* 没有特征提取优化的线性模型

```sh
$ python lm.py
```

* 包含特征提取优化的线性模型

```sh
$ python opt_lm.py
```

## 结果

| Weights averaged | Feature extracion optimized | precision |
| :--------------: | :-------------------------: | :-------: |
|        ×         |              ×              | 0.849818  |
|        ×         |              √              | 0.855283  |
|        √         |              ×              | 0.852581  |
|        √         |              √              | 0.856197  |

