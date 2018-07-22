# -*- coding: utf-8 -*-

import argparse
from datetime import datetime, timedelta

import numpy as np

from config import Config
from hmm import HMM, preprocess

# 解析命令参数
parser = argparse.ArgumentParser(
    description='Create Hidden Markov Model(HMM) for POS Tagging.'
)
parser.add_argument('-b', action='store_true', default=False,
                    dest='bigdata', help='use big data')
parser.add_argument('--file', '-f', action='store', dest='file',
                    help='set where to store the model')
args = parser.parse_args()

# 根据参数读取配置
config = Config(args.bigdata)

train = preprocess(config.ftrain)
dev = preprocess(config.fdev)
file = args.file if args.file else config.hmmpkl

wordseqs, tagseqs = zip(*train)
words, tags = sorted(set(np.hstack(wordseqs))), sorted(set(np.hstack(tagseqs)))

start = datetime.now()

print("Creating HMM with %d words and %d tags" % (len(words), len(tags)))
hmm = HMM(words, tags)

print("Using %d sentences to train the HMM" % (len(train)))
hmm.train(train, file)

print("Using Viterbi algorithm to tag the dataset")
tp, total, precision = hmm.evaluate(dev)
print("Precision of dev: %d / %d = %4f" % (tp, total, precision))

if args.bigdata:
    test = preprocess(config.ftest)
    hmm = HMM.load(file)
    tp, total, precision = hmm.evaluate(test)
    print("Precision of test: %d / %d = %4f" % (tp, total, precision))

print("%ss elapsed\n" % (datetime.now() - start))
