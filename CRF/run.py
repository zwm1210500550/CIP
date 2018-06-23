# -*- coding: utf-8 -*-

import argparse
import time

import numpy as np

from config import Config

# 解析命令参数
parser = argparse.ArgumentParser(
    description='Create Conditional Random Field(CRF) for POS Tagging.'
)
parser.add_argument('-b', action='store_true', default=False,
                    dest='bigdata', help='use big data')
parser.add_argument('--optimize', '-o', action='store_true', default=False,
                    dest='optimize', help='use feature extracion optimization')
parser.add_argument('--shuffle', '-s', action='store_true', default=False,
                    dest='shuffle', help='shuffle the data at each epoch')
args = parser.parse_args()

if args.optimize:
    from ocrf import CRF, preprocess
else:
    from crf import CRF, preprocess

# 根据参数读取配置
config = Config(args.bigdata)

train = preprocess(config.ftrain)
dev = preprocess(config.fdev)

all_words, all_tags = zip(*np.vstack(train))
tags = sorted(set(all_tags))

start = time.time()

print("Creating Conditional Random Field with %d tags" % (len(tags)))
crf = CRF(tags)

print("Using %d sentences to create the feature space" % (len(train)))
crf.create_feature_space(train)
print("The size of the feature space is %d" % crf.d)

print("Using SGD algorithm to train the model")
crf.SGD(train, dev, config.crfpkl,
        epochs=config.epochs,
        batch_size=config.batch_size,
        c=config.c,
        eta=config.eta,
        interval=config.interval,
        shuffle=args.shuffle)

if args.bigdata:
    test = preprocess(config.ftest)
    crf = CRF.load(config.crfpkl)
    print("Precision of test: %d / %d = %4f" % crf.evaluate(test))

print("%4fs elapsed\n" % (time.time() - start))
