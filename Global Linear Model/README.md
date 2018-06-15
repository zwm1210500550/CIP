# Global Linear Model

## Run

```sh
# 没有特征提取优化且没有特征累加
$ python glm.py
# 没有特征提取优化且包含特征累加
$ python glm.py average
# 包含特征提取优化且没有特征累加
$ python oglm.py
# 包含特征提取优化且包含特征累加
$ python oglm.py average
```

## Results

| Averaged Perceptron | Feature extracion optimization | precision |
| :-----------------: | :----------------------------: | :-------: |
|          ×          |               ×                | 0.866532  |
|          ×          |               √                | 0.873169  |
|          √          |               ×                | 0.875296  |
|          √          |               √                | 0.882291  |