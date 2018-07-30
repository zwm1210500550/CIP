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
│   ├── allm.txt
│   ├── asllm.txt
│   ├── llm.txt
│   ├── oallm.txt
│   ├── oasllm.txt
│   ├── ollm.txt
│   ├── osllm.txt
│   └── sllm.txt
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
|      ×       |    ×     |    ×     |  68/79   | 93.2983% | 92.9577% | 0:13:10.774230 |
|      ×       |    ×     |    √     |  19/30   | 93.7253% | 93.4651% | 0:13:45.439839 |
|      ×       |    √     |    ×     |  37/48   | 93.4401% | 93.0888% | 0:13:12.867444 |
|      ×       |    √     |    √     |  16/27   | 93.9071% | 93.6417% | 0:13:43.845564 |
|      √       |    ×     |    ×     |  46/57   | 93.3967% | 93.0802% | 0:02:26.644951 |
|      √       |    ×     |    √     |  16/27   | 93.8771% | 93.6159% | 0:02:31.493213 |
|      √       |    √     |    ×     |  38/49   | 93.4901% | 93.1808% | 0:02:29.142001 |
|      √       |    √     |    √     |  27/38   | 93.9288% | 93.6968% | 0:02:36.570588 |