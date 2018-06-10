# LinearModel

```sh
$ python lm.py
Creating Linear Model with 4537 words and 31 tags
Using 803 sentences to create the feature space
The size of the feature space is 81113
Using online-training algorithm to train the model
iteration 0: 18525 / 20454 = 0.905691
iteration 1: 18984 / 20454 = 0.928131
iteration 2: 19571 / 20454 = 0.956830
iteration 3: 19911 / 20454 = 0.973453
iteration 4: 20134 / 20454 = 0.984355
iteration 5: 20231 / 20454 = 0.989097
iteration 6: 20253 / 20454 = 0.990173
iteration 7: 20301 / 20454 = 0.992520
iteration 8: 20375 / 20454 = 0.996138
iteration 9: 20396 / 20454 = 0.997164
iteration 10: 20428 / 20454 = 0.998729
iteration 11: 20422 / 20454 = 0.998436
iteration 12: 20445 / 20454 = 0.999560
iteration 13: 20447 / 20454 = 0.999658
iteration 14: 20452 / 20454 = 0.999902
iteration 15: 20454 / 20454 = 1.000000
iteration 16: 20454 / 20454 = 1.000000
iteration 17: 20454 / 20454 = 1.000000
iteration 18: 20454 / 20454 = 1.000000
iteration 19: 20454 / 20454 = 1.000000
Using the trained model to tag 1910 sentences
Evaluating the result
precision: 42762 / 50319 = 0.849818
195.531896s elapsed
$ python optimized_lm.py
Creating Linear Model with 4537 words and 31 tags
Using 803 sentences to create the feature space
The size of the feature space is 67359
Using online-training algorithm to train the model
iteration 0: 18754 / 20454 = 0.916887
iteration 1: 19765 / 20454 = 0.966315
iteration 2: 20013 / 20454 = 0.978439
iteration 3: 20177 / 20454 = 0.986457
iteration 4: 20171 / 20454 = 0.986164
iteration 5: 20331 / 20454 = 0.993987
iteration 6: 20414 / 20454 = 0.998044
iteration 7: 20417 / 20454 = 0.998191
iteration 8: 20395 / 20454 = 0.997115
iteration 9: 20289 / 20454 = 0.991933
iteration 10: 20296 / 20454 = 0.992275
iteration 11: 20454 / 20454 = 1.000000
iteration 12: 20454 / 20454 = 1.000000
iteration 13: 20454 / 20454 = 1.000000
iteration 14: 20454 / 20454 = 1.000000
iteration 15: 20454 / 20454 = 1.000000
iteration 16: 20454 / 20454 = 1.000000
iteration 17: 20454 / 20454 = 1.000000
iteration 18: 20454 / 20454 = 1.000000
iteration 19: 20454 / 20454 = 1.000000
Using the trained model to tag 1910 sentences
Evaluating the result
precision: 43037 / 50319 = 0.855283
25.894133s elapsed
```


