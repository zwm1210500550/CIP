# Global Linear Model

## 结构

```sh
.
├── bigdata
│   ├── dev.conll
│   ├── test.conll
│   └── train.conll
├── data
│   ├── dev.conll
│   └── train.conll
├── result
│   ├── aglm.txt
│   ├── asglm.txt
│   ├── glm.txt
│   ├── oaglm.txt
│   ├── oasglm.txt
│   ├── oglm.txt
│   ├── osglm.txt
│   └── sglm.txt
├── config.py
├── glm.py
├── oglm.py
├── README.md
└── run.py
```

## 用法

```sh
$ python run.py -h
usage: run.py [-h] [-b] [--average] [--optimize] [--shuffle]

Create Global Linear Model(GLM) for POS Tagging.

optional arguments:
  -h, --help      show this help message and exit
  -b              use big data
  --average, -a   use average perceptron
  --optimize, -o  use feature extracion optimization
  --shuffle, -s   shuffle the data at each epoch
  --file FILE, -f FILE  set where to store the model
```

## 结果

### 小数据集

| 特征提取优化 | 权重累加 | 打乱数据 | 迭代次数 |  dev/P   | test/P |     mT(s)      |
| :----------: | :------: | :------: | :------: | :------: | :----: | :------------: |
|      ×       |    ×     |    ×     |  16/22   | 86.6532% |   *    | 0:00:21.914604 |
|      ×       |    ×     |    √     |  16/22   | 87.0288% |   *    | 0:00:22.851137 |
|      ×       |    √     |    ×     |  13/19   | 87.4839% |   *    | 0:00:22.770413 |
|      ×       |    √     |    √     |  17/23   | 87.7243% |   *    | 0:00:22.012788 |
|      √       |    ×     |    ×     |  14/20   | 87.3169% |   *    | 0:00:04.862168 |
|      √       |    ×     |    √     |   9/15   | 87.4262% |   *    | 0:00:04.941748 |
|      √       |    √     |    ×     |  20/26   | 88.0741% |   *    | 0:00:04.753132 |
|      √       |    √     |    √     |  11/17   | 88.2748% |   *    | 0:00:04.877338 |

### 大数据集

| 特征提取优化 | 权重累加 | 打乱数据 | 迭代次数 |  dev/P   |  test/P  |     mT(s)      |
| :----------: | :------: | :------: | :------: | :------: | :------: | :------------: |
|      ×       |    ×     |    ×     |  36/47   | 93.0898% | 92.7321% | 0:13:08.757926 |
|      ×       |    ×     |    √     |  19/30   | 93.4501% | 93.0251% | 0:13:38.431069 |
|      ×       |    √     |    ×     |  23/34   | 93.9905% | 93.7716% | 0:13:24.307518 |
|      ×       |    √     |    √     |  13/24   | 94.2624% | 94.0241% | 0:13:40.348989 |
|      √       |    ×     |    ×     |  36/47   | 93.2483% | 92.9699% | 0:02:33.052827 |
|      √       |    ×     |    √     |  19/30   | 93.7120% | 93.3438% | 0:02:44.118505 |
|      √       |    √     |    ×     |   8/19   | 94.0872% | 93.8660% | 0:02:52.517310 |
|      √       |    √     |    √     |   8/19   | 94.2874% | 94.1087% | 0:02:55.703809 |