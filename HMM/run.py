# -*- coding: utf-8 -*-

import argparse
import time

import numpy as np

from config import Config
from hmm import HMM, preprocess

# 解析命令参数
parser = argparse.ArgumentParser(
    description='Create Hidden Markov Model(HMM) for POS Tagging.'
)
parser.add_argument('-b', action='store_true', default=False,
                    dest='bigdata', help='use big data')
args = parser.parse_args()

# 根据参数读取配置
config = Config(args.bigdata)

train = preprocess(config.ftrain)
dev = preprocess(config.fdev)

all_words, all_tags = zip(*np.vstack(train))
words, tags = sorted(set(all_words)), sorted(set(all_tags))

start = time.time()

print("Creating HMM with %d words and %d tags" % (len(words), len(tags)))
hmm = HMM(words, tags)

print("Using %d sentences to train the HMM" % (len(train)))
hmm.train(train, file=config.hmmpkl)

print("Using Viterbi algorithm to tag the dataset")
tp, total, precision = hmm.evaluate(dev)
print("Precision of dev: %d / %d = %4f" % (tp, total, precision))

if args.bigdata:
    test = preprocess(config.ftest)
    hmm = HMM.load(config.hmmpkl)
    tp, total, precision = hmm.evaluate(test)
    print("Precision of test: %d / %d = %4f" % (tp, total, precision))

print("%4fs elapsed\n" % (time.time() - start))
