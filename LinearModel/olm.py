# -*- coding: utf-8 -*-

import pickle
import random
import time

import numpy as np


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


class LinearModel(object):

    def __init__(self, tags):
        # 所有不同的词性
        self.tags = tags
        # 词性对应索引的字典
        self.tdict = {t: i for i, t in enumerate(tags)}

        self.n = len(self.tags)

    def create_feature_space(self, sentences):
        # 特征空间
        self.epsilon = list({
            f for wordseq, tagseq in sentences
            for i, tag in enumerate(tagseq)
            for f in self.instantiate(wordseq, i)
        })
        # 特征对应索引的字典
        self.fdict = {f: i for i, f in enumerate(self.epsilon)}
        # 特征空间维度
        self.d = len(self.epsilon)

        # 特征权重
        self.W = np.zeros((self.d, self.n))
        # 累加特征权重
        self.V = np.zeros((self.d, self.n))

    def online(self, train, dev, file, epochs, interval, average, shuffle):
        max_e, max_precision = 0, 0.0
        # 迭代指定次数训练模型
        for epoch in range(epochs):
            start = time.time()
            # 随机打乱数据
            if shuffle:
                random.shuffle(train)
            # 保存更新时间戳和每个特征最近更新时间戳的记录
            self.k, self.R = 0, np.zeros((self.d, self.n), dtype='int')
            for batch in train:
                self.update(batch)
            self.V += [(self.k - r) * w for r, w in zip(self.R, self.W)]

            print("Epoch %d / %d: " % (epoch, epochs))
            result = self.evaluate(train, average=average)
            print("\ttrain: %d / %d = %4f" % result)
            tp, total, precision = self.evaluate(dev, average=average)
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

    def update(self, batch):
        wordseq, tagseq = batch
        # 根据单词序列的正确词性更新权重
        for i, tag in enumerate(tagseq):
            # 根据现有权重向量预测词性
            pre = self.predict(wordseq, i)
            # 如果预测词性与正确词性不同，则更新权重
            if tag != pre:
                fv = self.instantiate(wordseq, i)
                ti, pi = self.tdict[tag], self.tdict[pre]
                fis = [self.fdict[f] for f in fv if f in self.fdict]
                for fi in fis:
                    prev_w, prev_r = self.W[fi, [ti, pi]], self.R[fi, [ti, pi]]
                    # 累加权重加上步长乘以权重
                    self.V[fi, [ti, pi]] += (self.k - prev_r) * prev_w
                    # 更新权重
                    self.W[fi, [ti, pi]] += [1, -1]
                    # 更新时间戳记录
                    self.R[fi, [ti, pi]] = self.k
                self.k += 1

    def predict(self, wordseq, index, average=False):
        fv = self.instantiate(wordseq, index)
        scores = self.score(fv, average=average)
        return self.tags[np.argmax(scores)]

    def score(self, fvector, average=False):
        # 计算特征对应累加权重的得分
        if average:
            scores = np.array([self.V[self.fdict[f]]
                               for f in fvector if f in self.fdict])
        # 计算特征对应未累加权重的得分
        else:
            scores = np.array([self.W[self.fdict[f]]
                               for f in fvector if f in self.fdict])
        return np.sum(scores, axis=0)

    def instantiate(self, wordseq, index):
        word = wordseq[index]
        prev_word = wordseq[index - 1] if index > 0 else "^^"
        next_word = wordseq[index + 1] if index < len(wordseq) - 1 else "$$"
        prev_char = prev_word[-1]
        next_char = next_word[0]
        first_char = word[0]
        last_char = word[-1]

        fvector = []
        fvector.append(('02', word))
        fvector.append(('03', prev_word))
        fvector.append(('04', next_word))
        fvector.append(('05', word, prev_char))
        fvector.append(('06', word, next_char))
        fvector.append(('07', first_char))
        fvector.append(('08', last_char))

        for char in word[1:-1]:
            fvector.append(('09', char))
            fvector.append(('10', first_char, char))
            fvector.append(('11', last_char, char))
        if len(word) == 1:
            fvector.append(('12', word, prev_char, next_char))
        for i in range(1, len(word)):
            prev_char, char = word[i - 1], word[i]
            if prev_char == char:
                fvector.append(('13', char, 'consecutive'))
            if i <= 4:
                fvector.append(('14', word[:i]))
                fvector.append(('15', word[-i:]))
        if len(word) <= 4:
            fvector.append(('14', word))
            fvector.append(('15', word))
        return fvector

    def evaluate(self, sentences, average=False):
        tp, total = 0, 0

        for wordseq, tagseq in sentences:
            total += len(wordseq)
            preseq = np.array([self.predict(wordseq, i, average)
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
