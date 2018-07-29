# Linear Model

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
│   ├── alm.txt
│   ├── aslm.txt
│   ├── lm.txt
│   ├── oalm.txt
│   ├── oaslm.txt
│   ├── olm.txt
│   ├── oslm.txt
│   └── slm.txt
├── config.py
├── lm.py
├── olm.py
├── README.md
└── run.py
```

## 用法

```sh
$ python run.py -h
usage: run.py [-h] [-b] [--average] [--optimize] [--shuffle]

Create Linear Model(LM) for POS Tagging.

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
|      ×       |    ×     |    ×     |  14/20   | 84.4790% |   *    | 0:00:19.109967 |
|      ×       |    ×     |    √     |   4/10   | 84.9003% |   *    | 0:00:17.441059 |
|      ×       |    √     |    ×     |  19/25   | 85.4528% |   *    | 0:00:17.975325 |
|      ×       |    √     |    √     |  14/20   | 85.7907% |   *    | 0:00:19.258782 |
|      √       |    ×     |    ×     |   8/14   | 85.4588% |   *    | 0:00:02.118205 |
|      √       |    ×     |    √     |   4/10   | 85.7748% |   *    | 0:00:02.284357 |
|      √       |    √     |    ×     |  13/19   | 85.7052% |   *    | 0:00:02.171389 |
|      √       |    √     |    √     |   9/15   | 85.7986% |   *    | 0:00:02.333335 |

### 大数据集

| 特征提取优化 | 权重累加 | 打乱数据 | 迭代次数 |  dev/P   |  test/P  |     mT(s)      |
| :----------: | :------: | :------: | :------: | :------: | :------: | :------------: |
|      ×       |    ×     |    ×     |  31/42   | 92.3309% | 91.9108% | 0:09:47.356156 |
|      ×       |    ×     |    √     |  10/21   | 92.7796% | 92.5850% | 0:10:11.035721 |
|      ×       |    √     |    ×     |  33/44   | 93.5218% | 93.2580% | 0:09:40.645639 |
|      ×       |    √     |    √     |  11/22   | 93.8671% | 93.6245% | 0:10:09.906283 |
|      √       |    ×     |    ×     |  32/43   | 92.7362% | 92.2479% | 0:01:03.565462 |
|      √       |    ×     |    √     |  19/30   | 93.1432% | 92.8081% | 0:00:59.265025 |
|      √       |    √     |    ×     |  35/46   | 93.7420% | 93.4247% | 0:00:57.788527 |
|      √       |    √     |    √     |  17/28   | 93.9538% | 93.6343% | 0:00:57.509398 |