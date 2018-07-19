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
  --file FILE, -f FILE  set where to store the model
```

## 结果

### 大数据集

| 特征提取优化 | 模拟退火 | 打乱数据 | 迭代次数 |  dev/P   |  test/P  |     mT(s)      |
| :----------: | :------: | :------: | :------: | :------: | :------: | :------------: |
|      ×       |    ×     |    ×     |  68/79   | 93.2983% | 92.9577% | 0:15:43.232115 |
|      ×       |    ×     |    √     |  17/28   | 93.8204% | 93.5681% | 0:16:48.171849 |
|      ×       |    √     |    ×     |  37/48   | 93.4401% | 93.0888% | 0:15:57.349092 |
|      ×       |    √     |    √     |  18/29   | 93.9104% | 93.6073% | 0:15:54.446139 |
|      √       |    ×     |    ×     |  46/57   | 93.3967% | 93.0802% | 0:02:47.118654 |
|      √       |    ×     |    √     |  13/24   | 93.8771% | 93.6539% | 0:03:13.408598 |
|      √       |    √     |    ×     |  38/49   | 93.4901% | 93.1808% | 0:02:46.760723 |
|      √       |    √     |    √     |  11/22   | 93.9321% | 93.6588% | 0:02:56.124429 |