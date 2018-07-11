# -*- coding: utf-8 -*-

import argparse
from datetime import datetime, timedelta

import numpy as np

from config import Config

# 解析命令参数
parser = argparse.ArgumentParser(
    description='Create Log Linear Model(LLM) for POS Tagging.'
)
parser.add_argument('-b', action='store_true', default=False,
                    dest='bigdata', help='use big data')
parser.add_argument('--anneal', '-a', action='store_true', default=False,
                    dest='anneal', help='use simulated annealing')
parser.add_argument('--optimize', '-o', action='store_true', default=False,
                    dest='optimize', help='use feature extracion optimization')
parser.add_argument('--regularize', '-r', action='store_true', default=False,
                    dest='regularize', help='use L2 regularization')
parser.add_argument('--shuffle', '-s', action='store_true', default=False,
                    dest='shuffle', help='shuffle the data at each epoch')
args = parser.parse_args()

if args.optimize:
    from ollm import LogLinearModel, preprocess
else:
    from llm import LogLinearModel, preprocess

# 根据参数读取配置
config = Config(args.bigdata)

train = preprocess(config.ftrain)
dev = preprocess(config.fdev)

wordseqs, tagseqs = zip(*train)
tags = sorted(set(np.hstack(tagseqs)))

start = datetime.now()

print("Creating Log Linear Model with %d tags" % (len(tags)))
if args.optimize:
    print("\tuse feature extracion optimization")
if args.anneal:
    print("\tuse simulated annealing")
if args.regularize:
    print("\tuse L2 regularization")
if args.shuffle:
    print("\tshuffle the data at each epoch")
llm = LogLinearModel(tags)

print("Using %d sentences to create the feature space" % (len(train)))
llm.create_feature_space(train)
print("The size of the feature space is %d" % llm.d)

print("Using SGD algorithm to train the model")
print("\tepochs: %d\n\tbatch_size: %d\n\tinterval: %d" %
      (config.epochs, config.batch_size,  config.interval))
if args.anneal:
    print("\teta: %f\n\tdacay: %f" % (config.eta, config.decay))
if args.regularize:
    print("\tc: %f" % config.c)
llm.SGD(train, dev, config.llmpkl,
        epochs=config.epochs,
        batch_size=config.batch_size,
        interval=config.interval,
        c=config.c,
        eta=config.eta,
        decay=config.decay,
        anneal=args.anneal,
        regularize=args.regularize,
        shuffle=args.shuffle)

if args.bigdata:
    test = preprocess(config.ftest)
    llm = LogLinearModel.load(config.llmpkl)
    print("Precision of test: %d / %d = %4f" % llm.evaluate(test))

print("%ss elapsed\n" % (datetime.now() - start))
