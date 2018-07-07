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
```

## 结果

### 小数据集

| 特征提取优化 | 权重累加 | 打乱数据 | 迭代次数 |  dev/P   | test/P |   time(s)    |
| :----------: | :------: | :------: | :------: | :------: | :----: | :----------: |
|      ×       |    ×     |    ×     |  16/22   | 86.6532% |   *    | 11687.483856 |
|      ×       |    ×     |    √     |  15/21   | 86.7088% |   *    | 11306.076008 |
|      ×       |    √     |    ×     |  13/19   | 87.4839% |   *    | 10359.376148 |
|      ×       |    √     |    √     |  19/25   | 87.7005% |   *    | 13454.770436 |
|      √       |    ×     |    ×     |  14/20   | 87.3169% |   *    | 1170.845839  |
|      √       |    ×     |    √     |  17/23   | 87.6309% |   *    | 1399.469547  |
|      √       |    √     |    ×     |  20/26   | 88.0741% |   *    | 1501.862312  |
|      √       |    √     |    √     |  14/20   | 88.0443% |   *    | 1185.835210  |

### 大数据集

| 特征提取优化 | 权重累加 | 打乱数据 | 迭代次数 |  dev/P   |  test/P  |    time(s)    |
| :----------: | :------: | :------: | :------: | :------: | :------: | :-----------: |
|      ×       |    ×     |    ×     |    /     |    %     |    %     |               |
|      ×       |    ×     |    √     |  15/26   | 93.4301% | 93.0668% | 472424.963864 |
|      ×       |    √     |    ×     |    /     |    %     |    %     |               |
|      ×       |    √     |    √     |  15/26   | 94.2223% | 92.9258% | 488158.716056 |
|      √       |    ×     |    ×     |  36/47   | 93.2483% | 92.9699% | 78205.660869  |
|      √       |    ×     |    √     |  20/31   | 93.5869% | 93.4210% | 56210.397132  |
|      √       |    √     |    ×     |   8/19   | 94.0872% | 92.1069% | 34005.065474  |
|      √       |    √     |    √     |  10/21   | 94.3441% | 92.8706% | 40073.499127  |