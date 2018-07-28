# -*- coding: utf-8 -*-

import argparse
from datetime import datetime, timedelta

import numpy as np

from config import Config
from corpus import Corpus
from hmm import HMM

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

print("Preprocessing the data")
corpus = Corpus(config.ftrain)
train = corpus.load(config.ftrain)
dev = corpus.load(config.fdev)
file = args.file if args.file else config.hmmpkl

start = datetime.now()

print("Creating HMM with %d words and %d tags" % (corpus.nw, corpus.nt))
hmm = HMM(corpus.nw, corpus.nt)

print("Using %d sentences to train the HMM" % (len(corpus.sentences)))
hmm.train(train, file)

print("Using Viterbi algorithm to tag the dataset")
tp, total, precision = hmm.evaluate(dev)
print("Precision of dev: %d / %d = %4f" % (tp, total, precision))

if args.bigdata:
    hmm = HMM.load(file)
    test = corpus.load(config.ftest)
    tp, total, precision = hmm.evaluate(test)
    print("Precision of test: %d / %d = %4f" % (tp, total, precision))

print("%ss elapsed\n" % (datetime.now() - start))
