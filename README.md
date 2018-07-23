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
|      ×       |    ×     |    ×     |  14/20   | 84.4790% |   *    | 0:00:21.023148 |
|      ×       |    ×     |    √     |  12/18   | 84.8805% |   *    | 0:00:20.990117 |
|      ×       |    √     |    ×     |  19/25   | 85.4528% |   *    | 0:00:22.436925 |
|      ×       |    √     |    √     |  14/20   | 85.7569% |   *    | 0:00:23.145980 |
|      √       |    ×     |    ×     |   8/14   | 85.4588% |   *    | 0:00:02.478257 |
|      √       |    ×     |    √     |   8/14   | 85.7191% |   *    | 0:00:02.598164 |
|      √       |    √     |    ×     |  13/19   | 85.7052% |   *    | 0:00:02.456955 |
|      √       |    √     |    √     |   7/13   | 85.8026% |   *    | 0:00:02.530138 |

### 大数据集

| 特征提取优化 | 权重累加 | 打乱数据 | 迭代次数 |  dev/P   |  test/P  |     mT(s)      |
| :----------: | :------: | :------: | :------: | :------: | :------: | :------------: |
|      ×       |    ×     |    ×     |  31/42   | 92.3309% | 91.9108% | 0:14:47.007310 |
|      ×       |    ×     |    √     |  30/41   | 92.9013% | 92.6169% | 0:12:13.890569 |
|      ×       |    √     |    ×     |  33/44   | 93.5218% | 93.2580% | 0:13:35.569657 |
|      ×       |    √     |    √     |  11/22   | 937920%  | 93.5313% | 0:12:16.117138 |
|      √       |    ×     |    ×     |  32/43   | 92.7362% | 92.2479% | 0:01:08.270188 |
|      √       |    ×     |    √     |  25/36   | 93.1549% | 92.7051% | 0:01:11.968326 |
|      √       |    √     |    ×     |  35/46   | 93.7420% | 93.4247% | 0:01:10.496375 |
|      √       |    √     |    √     |  17/28   | 93.8637% | 93.6821% | 0:01:15.179984 |