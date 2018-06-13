# Linear Model

## Run

```sh
# 没有特征提取优化且没有特征累加
$ python lm.py
# 没有特征提取优化且包含特征累加
$ python lm.py average
# 包含特征提取优化且没有特征累加
$ python olm.py
# 包含特征提取优化且包含特征累加
$ python olm.py average
```

## Results

| Averaged Perceptron | Feature extracion optimization | precision |
| :-----------------: | :----------------------------: | :-------: |
|          ×          |               ×                | 0.844790  |
|          ×          |               √                | 0.854588  |
|          √          |               ×                | 0.854528  |
|          √          |               √                | 0.857052  |

