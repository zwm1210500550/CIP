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
  -b          use big data
  --file FILE, -f FILE  set where to store the model
```

## 结果

### 小数据集

| alpha |  dev/P   | test/P |     mT(s)      |
| :---: | :------: | :----: | :------------: |
|  0.3  | 74.2880% |   *    | 0:00:00.543569 |

### 大数据集

| alpha | dev/P | test/P | mT(s) |
| :------: | :-------: | :------: | :------: |
| 0.01  | 88.3546% | 88.4994%  | 0:00:02.734318 |