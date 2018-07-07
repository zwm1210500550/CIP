# -*- coding: utf-8 -*-

import pickle
import random
import time
from collections import defaultdict

import numpy as np
from scipy.misc import logsumexp


def preprocess(fdata):
    start = 0
    sentences = []
    with open(fdata, 'r') as train:
        lines = [line for line in train]
    for i, line in enumerate(lines):
        if len(lines[i]) <= 1:
            wordseq, tagseq = zip(*[l.split()[1:4:2] for l in lines[start:i]])
            start = i + 1
            while start < len(lines) and len(lines[start]) <= 1:
                start += 1
            sentences.append((wordseq, tagseq))
    return sentences


class LogLinearModel(object):

    def __init__(self, tags):
        # 所有不同的词性
        self.tags = tags

        self.n = len(self.tags)

    def create_feature_space(self, sentences):
        # 特征空间
        self.epsilon = list({
            f for wordseq, tagseq in sentences
            for i, tag in enumerate(tagseq)
            for f in self.instantiate(wordseq, i, tag)
        })
        # 特征对应索引的字典
        self.fdict = {f: i for i, f in enumerate(self.epsilon)}
        # 特征空间维度
        self.d = len(self.epsilon)

        # 特征权重
        self.W = np.zeros(self.d)

    def SGD(self, train, dev, file,
            epochs, batch_size, c, eta, decay, interval,
            anneal, regularize, shuffle):
        max_e, max_precision = 0, 0.0
        training_data = [(wordseq, i, tag)
                         for wordseq, tagseq in train
                         for i, tag in enumerate(tagseq)]
        # 迭代指定次数训练模型
        for epoch in range(epochs):
            start = time.time()
            # 随机打乱数据
            if shuffle:
                random.shuffle(training_data)
            # 设置L2正则化系数
            if not regularize:
                c = 0
            # 设置学习速率衰减
            eta = 1 / (1 + decay * epoch) * eta if anneal else 1
            # 按照指定大小对数据分割批次
            batches = [training_data[i:i + batch_size]
                       for i in range(0, len(training_data), batch_size)]
            # 根据批次数据更新权重
            for batch in batches:
                self.update(batch, c, eta)

            print("Epoch %d / %d: " % (epoch, epochs))
            print("\ttrain: %d / %d = %4f" % self.evaluate(train))
            tp, total, precision = self.evaluate(dev)
            print("\tdev: %d / %d = %4f" % (tp, total, precision))
            print("\t%4fs elapsed" % (time.time() - start))
            # 保存效果最好的模型
            if precision > max_precision:
                self.dump(file)
                max_e, max_precision = epoch, precision
            elif epoch - max_e > interval:
                break
        print("max precision of dev is %4f at epoch %d" %
              (max_precision, max_e))

    def update(self, batch, c, eta):
        gradients = defaultdict(float)

        for wordseq, i, tag in batch:
            fv = self.instantiate(wordseq, i, tag)
            fis = [self.fdict[f] for f in fv if f in self.fdict]
            for fi in fis:
                gradients[fi] += 1

            # 获取每个词性对应的所有特征
            fvs = [self.instantiate(wordseq, i, tag) for tag in self.tags]
            scores = [self.score(fv) for fv in fvs]
            probs = np.exp(scores - logsumexp(scores))

            for fv, p in zip(fvs, probs):
                fis = [self.fdict[f] for f in fv if f in self.fdict]
                for fi in fis:
                    gradients[fi] -= p

        if c != 0:
            self.W *= (1 - eta * c)
        for fi, v in gradients.items():
            self.W[fi] += eta * v

    def predict(self, wordseq, index):
        fvs = [self.instantiate(wordseq, index, tag)
               for tag in self.tags]
        scores = [self.score(fv) for fv in fvs]
        return self.tags[np.argmax(scores)]

    def score(self, fvector):
        scores = [self.W[self.fdict[f]]
                  for f in fvector if f in self.fdict]
        return sum(scores)

    def instantiate(self, wordseq, index, tag):
        word = wordseq[index]
        prev_word = wordseq[index - 1] if index > 0 else '^^'
        next_word = wordseq[index + 1] if index < len(wordseq) - 1 else '$$'
        prev_char = prev_word[-1]
        next_char = next_word[0]
        first_char = word[0]
        last_char = word[-1]

        fvector = []
        fvector.append(('02', tag, word))
        fvector.append(('03', tag, prev_word))
        fvector.append(('04', tag, next_word))
        fvector.append(('05', tag, word, prev_char))
        fvector.append(('06', tag, word, next_char))
        fvector.append(('07', tag, first_char))
        fvector.append(('08', tag, last_char))

        for char in word[1:-1]:
            fvector.append(('09', tag, char))
            fvector.append(('10', tag, first_char, char))
            fvector.append(('11', tag, last_char, char))
        if len(word) == 1:
            fvector.append(('12', tag, word, prev_char, next_char))
        for i in range(1, len(word)):
            prev_char, char = word[i - 1], word[i]
            if prev_char == char:
                fvector.append(('13', tag, char, 'consecutive'))
            if i <= 4:
                fvector.append(('14', tag, word[:i]))
                fvector.append(('15', tag, word[-i:]))
        if len(word) <= 4:
            fvector.append(('14', tag, word))
            fvector.append(('15', tag, word))
        return fvector

    def evaluate(self, sentences):
        tp, total = 0, 0

        for wordseq, tagseq in sentences:
            total += len(wordseq)
            preseq = np.array([self.predict(wordseq, i)
                               for i in range(len(wordseq))])
            tp += np.sum(tagseq == preseq)
        precision = tp / total
        return tp, total, precision

    def dump(self, file):
        with open(file, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, file):
        with open(file, 'rb') as f:
            hmm = pickle.load(f)
        return hmm
