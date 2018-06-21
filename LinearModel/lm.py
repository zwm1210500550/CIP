# -*- coding: utf-8 -*-

import sys
import time

import numpy as np


def preprocess(fdata):
    start = 0
    sentences = []
    with open(fdata, 'r') as train:
        lines = [line for line in train]
    for i, line in enumerate(lines):
        if len(lines[i]) <= 1:
            sentences.append([l.split()[1:4:2] for l in lines[start:i]])
            start = i + 1
            while start < len(lines) and len(lines[start]) <= 1:
                start += 1
    return sentences


class LinearModel(object):

    def __init__(self, tags):
        # 所有不同的词性
        self.tags = tags

        self.N = len(self.tags)

    def create_feature_space(self, sentences):
        feature_space = set()
        for sentence in sentences:
            wordseq, tagseq = zip(*sentence)
            for i, tag in enumerate(tagseq):
                fvector = self.instantiate(wordseq, i, tag)
                feature_space.update(fvector)

        # 特征空间
        self.epsilon = list(feature_space)
        # 特征对应索引的字典
        self.fdict = {f: i for i, f in enumerate(self.epsilon)}
        # 特征空间维度
        self.D = len(self.epsilon)

        # 特征权重
        self.W = np.zeros(self.D, dtype='int')
        # 累加特征权重
        self.V = np.zeros(self.D, dtype='int')

    def online(self, sentences, epochs=20):
        for epoch in range(epochs):
            for sentence in sentences:
                wordseq, tagseq = zip(*sentence)
                # 根据单词序列的正确词性更新权重
                for i, tag in enumerate(tagseq):
                    self.update(wordseq, i, tag)
            yield epoch

    def update(self, wordseq, index, tag):
        # 根据现有权重向量预测词性
        pre = self.predict(wordseq, index)
        # 如果预测词性与正确词性不同，则更新权重
        if tag != pre:
            cfvector = self.instantiate(wordseq, index, tag)
            efvector = self.instantiate(wordseq, index, pre)
            for cf, ef in zip(cfvector, efvector):
                if cf in self.fdict:
                    self.W[self.fdict[cf]] += 1
                if ef in self.fdict:
                    self.W[self.fdict[ef]] -= 1
            self.V += self.W

    def predict(self, wordseq, index, average=False):
        fvectors = [self.instantiate(wordseq, index, tag)
                    for tag in self.tags]
        scores = [self.score(fv, average=average) for fv in fvectors]
        return self.tags[np.argmax(scores)]

    def score(self, fvector, average=False):
        # 计算特征对应累加权重的得分
        if average:
            scores = [self.V[self.fdict[f]]
                      for f in fvector if f in self.fdict]
        # 计算特征对应未累加权重的得分
        else:
            scores = [self.W[self.fdict[f]]
                      for f in fvector if f in self.fdict]
        return np.sum(scores)

    def instantiate(self, wordseq, index, tag):
        word = wordseq[index]
        prev_word = wordseq[index - 1] if index > 0 else "^^"
        next_word = wordseq[index + 1] if index < len(wordseq) - 1 else "$$"
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

    def evaluate(self, sentences, average=False):
        tp, total = 0, 0

        for sentence in sentences:
            total += len(sentence)
            wordseq, tagseq = zip(*sentence)
            preseq = [self.predict(wordseq, i, average)
                      for i in range(len(wordseq))]
            tp += sum([t == p for t, p in zip(tagseq, preseq)])
        precision = tp / total
        return tp, total, precision


if __name__ == '__main__':
    train = preprocess('data/train.conll')
    dev = preprocess('data/dev.conll')

    all_words, all_tags = zip(*np.vstack(train))
    tags = sorted(set(all_tags))

    start = time.time()

    print("Creating Linear Model with %d tags" % (len(tags)))
    lm = LinearModel(tags)

    print("Using %d sentences to create the feature space" % (len(train)))
    lm.create_feature_space(train)
    print("The size of the feature space is %d" % lm.D)

    average = len(sys.argv) > 1 and sys.argv[1] == 'average'
    evaluations = []

    print("Using online-training algorithm to train the model")
    for epoch in lm.online(train):
        print("Epoch %d" % epoch)
        result = lm.evaluate(train, average=average)
        print("\ttrain: %d / %d = %4f" % result)
        result = lm.evaluate(dev, average=average)
        print("\tdev: %d / %d = %4f" % result)
        evaluations.append(result)

    print("Successfully evaluated dev data using the model")
    print("Precision: %d / %d = %4f" % max(evaluations, key=lambda x: x[2]))
    print("%4fs elapsed" % (time.time() - start))
