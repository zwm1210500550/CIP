# Log Linear Model

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
│   ├── bllm.txt
│   ├── bollm.txt
│   ├── llm.txt
│   └── ollm.txt
├── config.py
├── llm.py
├── ollm.py
├── README.md
└── run.py
```

## 用法

```sh
usage: run.py [-h] [-b] [--anneal] [--optimize] [--regularize] [--shuffle]

Create Log Linear Model(LLM) for POS Tagging.

optional arguments:
  -h, --help        show this help message and exit
  -b                use big data
  --anneal, -a      use simulated annealing
  --optimize, -o    use feature extracion optimization
  --regularize, -r  use L2 regularization
  --shuffle, -s     shuffle the data at each epoch
```

## 结果

### 小数据集

| 特征提取优化 | 模拟退火 | L2正则化 | 打乱数据 | 迭代次数 | dev/P | test/P | time(s) |
| :----------: | :------: | :------: | :------: | :------: | :---: | :----: | :-----: |
|      ×       |    ×     |    ×     |    ×     |          |       |   *    |         |
|      ×       |    ×     |    ×     |    √     |          |       |   *    |         |
|      ×       |    ×     |    √     |    ×     |          |       |   *    |         |
|      ×       |    ×     |    √     |    √     |          |       |   *    |         |
|      ×       |    √     |    ×     |    ×     |          |       |   *    |         |
|      ×       |    √     |    ×     |    √     |          |       |   *    |         |
|      ×       |    √     |    √     |    ×     |          |       |   *    |         |
|      ×       |    √     |    √     |    √     |          |       |   *    |         |
|      √       |    ×     |    ×     |    ×     |          |       |   *    |         |
|      √       |    ×     |    ×     |    √     |          |       |   *    |         |
|      √       |    ×     |    √     |    ×     |          |       |   *    |         |
|      √       |    ×     |    √     |    √     |          |       |   *    |         |
|      √       |    √     |    ×     |    ×     |          |       |   *    |         |
|      √       |    √     |    ×     |    √     |          |       |   *    |         |
|      √       |    √     |    √     |    ×     |          |       |   *    |         |
|      √       |    √     |    √     |    √     |          |       |   *    |         |

### 大数据集

| 特征提取优化 | 模拟退火 | L2正则化 | 打乱数据 | 迭代次数 | dev/P | test/P | time(s) |
| :----------: | :------: | :------: | :------: | :------: | :---: | :----: | :-----: |
|      ×       |    ×     |    ×     |    ×     |          |       |        |         |
|      ×       |    ×     |    ×     |    √     |          |       |        |         |
|      ×       |    ×     |    √     |    ×     |          |       |        |         |
|      ×       |    ×     |    √     |    √     |          |       |        |         |
|      ×       |    √     |    ×     |    ×     |          |       |        |         |
|      ×       |    √     |    ×     |    √     |          |       |        |         |
|      ×       |    √     |    √     |    ×     |          |       |        |         |
|      ×       |    √     |    √     |    √     |          |       |        |         |
|      √       |    ×     |    ×     |    ×     |          |       |        |         |
|      √       |    ×     |    ×     |    √     |          |       |        |         |
|      √       |    ×     |    √     |    ×     |          |       |        |         |
|      √       |    ×     |    √     |    √     |          |       |        |         |
|      √       |    √     |    ×     |    ×     |          |       |        |         |
|      √       |    √     |    ×     |    √     |          |       |        |         |
|      √       |    √     |    √     |    ×     |          |       |        |         |
|      √       |    √     |    √     |    √     |          |       |        |         |