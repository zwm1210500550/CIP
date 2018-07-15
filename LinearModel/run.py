# -*- coding: utf-8 -*-

import argparse
from datetime import datetime, timedelta

import numpy as np

from config import Config

# 解析命令参数
parser = argparse.ArgumentParser(
    description='Create Linear Model(LM) for POS Tagging.'
)
parser.add_argument('-b', action='store_true', default=False,
                    dest='bigdata', help='use big data')
parser.add_argument('--average', '-a', action='store_true', default=False,
                    dest='average', help='use average perceptron')
parser.add_argument('--optimize', '-o', action='store_true', default=False,
                    dest='optimize', help='use feature extracion optimization')
parser.add_argument('--shuffle', '-s', action='store_true', default=False,
                    dest='shuffle', help='shuffle the data at each epoch')
parser.add_argument('--file', '-f', action='store', dest='file',
                    help='set where to store the model')
args = parser.parse_args()

if args.optimize:
    from olm import LinearModel, preprocess
else:
    from lm import LinearModel, preprocess

# 根据参数读取配置
config = Config(args.bigdata)

train = preprocess(config.ftrain)
dev = preprocess(config.fdev)
file = args.file if args.file else config.lmpkl

wordseqs, tagseqs = zip(*train)
tags = sorted(set(np.hstack(tagseqs)))

start = datetime.now()

print("Creating Linear Model with %d tags" % (len(tags)))
if args.optimize:
    print("\tuse feature extracion optimization")
if args.average:
    print("\tuse average perceptron")
if args.shuffle:
    print("\tshuffle the data at each epoch")
lm = LinearModel(tags)

print("Using %d sentences to create the feature space" % (len(train)))
lm.create_feature_space(train)
print("The size of the feature space is %d" % lm.d)

print("Using online-training algorithm to train the model")
print("\tepochs: %d\n\tinterval: %d" % (config.epochs, config.interval))
lm.online(train, dev, file,
          epochs=config.epochs,
          interval=config.interval,
          average=args.average,
          shuffle=args.shuffle)

if args.bigdata:
    test = preprocess(config.ftest)
    lm = LinearModel.load(file)
    print("Precision of test: %d / %d = %4f" %
          lm.evaluate(test, average=args.average))

print("%ss elapsed\n" % (datetime.now() - start))
