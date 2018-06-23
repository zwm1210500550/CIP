# -*- coding: utf-8 -*-

import argparse
import time

import numpy as np

from config import Config

# 解析命令参数
parser = argparse.ArgumentParser(
    description='Create Global Linear Model(GLM) for POS Tagging.'
)
parser.add_argument('-b', action='store_true', default=False,
                    dest='bigdata', help='use big data')
parser.add_argument('--average', '-a', action='store_true', default=False,
                    dest='average', help='use average perceptron')
parser.add_argument('--optimize', '-o', action='store_true', default=False,
                    dest='optimize', help='use feature extracion optimization')
parser.add_argument('--shuffle', '-s', action='store_true', default=False,
                    dest='shuffle', help='shuffle the data at each epoch')
args = parser.parse_args()

if args.optimize:
    from oglm import GlobalLinearModel, preprocess
else:
    from glm import GlobalLinearModel, preprocess

# 根据参数读取配置
config = Config(args.bigdata)

train = preprocess(config.ftrain)
dev = preprocess(config.fdev)

all_words, all_tags = zip(*np.vstack(train))
tags = sorted(set(all_tags))

start = time.time()

print("Creating Global Linear Model with %d tags" % (len(tags)))
glm = GlobalLinearModel(tags)

print("Using %d sentences to create the feature space" % (len(train)))
glm.create_feature_space(train)
print("The size of the feature space is %d" % glm.d)

print("Using online-training algorithm to train the model")
glm.online(train, dev, config.glmpkl,
           epochs=config.epochs,
           interval=config.interval,
           average=args.average,
           shuffle=args.shuffle)

if args.bigdata:
    test = preprocess(config.ftest)
    glm = GlobalLinearModel.load(config.glmpkl)
    print("Precision of test: %d / %d = %4f" % glm.evaluate(test))

print("%4fs elapsed\n" % (time.time() - start))
