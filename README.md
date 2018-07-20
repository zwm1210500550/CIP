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
|      ×       |    ×     |    ×     |  16/22   | 86.6532% |   *    | 0:00:28.035253 |
|      ×       |    ×     |    √     |  12/18   | 87.0109% |   *    | 0:00:30.352954 |
|      ×       |    √     |    ×     |  13/19   | 87.4839% |   *    | 0:00:28.448236 |
|      ×       |    √     |    √     |  13/19   | 87.8773% |   *    | 0:00:27.522421 |
|      √       |    ×     |    ×     |  14/20   | 87.3169% |   *    | 0:00:05.708704 |
|      √       |    ×     |    √     |  14/20   | 87.8873% |   *    | 0:00:06.073969 |
|      √       |    √     |    ×     |  20/26   | 88.0741% |   *    | 0:00:05.204048 |
|      √       |    √     |    √     |  11/17   | 88.1735% |   *    | 0:00:05.840512 |

### 大数据集

| 特征提取优化 | 权重累加 | 打乱数据 | 迭代次数 |  dev/P   |  test/P  |     mT(s)      |
| :----------: | :------: | :------: | :------: | :------: | :------: | :------------: |
|      ×       |    ×     |    ×     |  36/47   | 93.0898% | 92.8522% | 0:15:53.709974 |
|      ×       |    ×     |    √     |  12/23   | 93.3700% | 93.1391% | 0:15:55.060772 |
|      ×       |    √     |    ×     |  23/34   | 93.9905% | 93.7716% | 0:15:56.358708 |
|      ×       |    √     |    √     |  22/33   | 94.2073% | 94.0253% | 0:16:27.548416 |
|      √       |    ×     |    ×     |  36/47   | 93.2483% | 92.9699% | 0:03:27.127443 |
|      √       |    ×     |    √     |  29/40   | 93.6352% | 93.4688% | 0:03:13.687172 |
|      √       |    √     |    ×     |   8/19   | 94.0872% | 93.8660% | 0:03:24.776367 |
|      √       |    √     |    √     |   9/20   | 94.3408% | 94.1847% | 0:03:26.355785 |