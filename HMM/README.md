# Hidden Markov Model

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
│   └── hmm.txt
├── config.py
├── hmm.py
├── README.md
└── run.py
```

## 用法

```sh
$ python run.py -h
usage: run.py [-h] [-b]

Create Hidden Markov Model(HMM) for POS Tagging

optional arguments:
  -h, --help  show this help message and exit
  -b          Use big data
```

## 运行

```sh
# 小数据集
$ python run.py
# 大数据集
$ python run.py -b
```

## 结果

|  | alpha | dev/P(%) | test/P(%) | time(s)  |
| :---: | :------: | :-------: | :------: | :------: |
|  小数据集  |  0.3  | 74.2880% | * | 5.535537 |
| 大数据集 | 0.01  | 88.3546% | 88.4994%  | 17.214356 |