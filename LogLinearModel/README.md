# Log Linear Model

## Run

```sh
# 没有特征提取优化
$ python llm.py
# 包含特征提取优化
$ python ollm.py
```

## Results

| L2 regularization | Simulated annealing | Feature extracion optimization | precision |
| :---------------: | :-----------------: | :----------------------------: | :-------: |
|         ×         |          ×          |               ×                | 0.871222  |
|         ×         |          √          |               √                | 0.872652  |
|         √         |          √          |               ×                | 0.872831  |
|         √         |          √          |               √                | 0.874024  |

