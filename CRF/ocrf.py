# -*- coding: utf-8 -*-

import pickle
import time

import numpy as np
from scipy.misc import logsumexp


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


class CRF(object):

    def __init__(self, tags):
        # 所有不同的词性
        self.tags = tags
        # 句首词性
        self.BOS = 'BOS'
        # 句尾词性
        self.EOS = 'EOS'
        # 词性对应索引的字典
        self.tdict = {t: i for i, t in enumerate(tags)}

        self.n = len(self.tags)

    def create_feature_space(self, sentences):
        feature_space = set()
        for sentence in sentences:
            wordseq, tagseq = zip(*sentence)
            prev_tag = self.BOS
            for i, tag in enumerate(tagseq):
                fvector = self.instantiate(wordseq, i, prev_tag)
                feature_space.update(fvector)
                prev_tag = tag

        # 特征空间
        self.epsilon = list(feature_space)
        # 特征对应索引的字典
        self.fdict = {f: i for i, f in enumerate(self.epsilon)}
        # 特征空间维度
        self.d = len(self.epsilon)

        # 特征权重
        self.W = np.zeros((self.d, self.n))

    def SGD(self, train, dev, file, epochs, batch_size, c, eta,
            interval, shuffle):
        max_e, max_precision = 0, 0.0
        for epoch in range(epochs):
            start = time.time()
            # 随机打乱数据
            if shuffle:
                random.shuffle(train)
            batches = [train[i:i + batch_size]
                       for i in range(0, len(train), batch_size)]
            for batch in batches:
                # 根据批次数据更新权重
                self.update(batch, c, max(eta, 0.00001))
                # eta *= 0.999

            print("Epoch %d / %d: " % (epoch, epochs))
            result = self.evaluate(train)
            print("\ttrain: %d / %d = %4f" % result)
            tp, total, precision = self.evaluate(dev)
            print("\tdev: %d / %d = %4f" % (tp, total, precision))
            print("\t%4f elapsed" % (time.time() - start))
            if precision > max_precision:
                self.dump(file)
                max_e, max_precision = epoch, precision
            elif epoch - max_e > interval:
                break
        print("max precision of dev is %4f at epoch %d" %
              (max_precision, max_e))

    def update(self, batch, c, eta):
        gradients = np.zeros((self.d, self.n))
        for sentence in batch:
            wordseq, tagseq = zip(*sentence)

            prev_tag = self.BOS
            for i, tag in enumerate(tagseq):
                for f in self.instantiate(wordseq, i, prev_tag):
                    if f in self.fdict:
                        gradients[self.fdict[f], self.tdict[tag]] += 1
                prev_tag = tag

            alpha = self.forward(wordseq)
            beta = self.backward(wordseq)
            logZ = logsumexp(alpha[-1])
            # print(logZ, self.logsumexp(
            #     beta[0] + self.score(self.instantiate(wordseq, 0, self.BOS))))

            fvector = self.instantiate(wordseq, 0, self.BOS)
            p = np.exp(self.score(fvector) + beta[0] - logZ)
            for f in fvector:
                if f in self.fdict:
                    gradients[self.fdict[f]] -= p

            for i in range(1, len(tagseq)):
                fvectors = [self.instantiate(wordseq, i, prev_tag)
                            for prev_tag in self.tags]
                for j, fv in enumerate(fvectors):
                    score = self.score(fv)
                    p = np.exp(score + alpha[i - 1, j] + beta[i] - logZ)
                    for f in fv:
                        if f in self.fdict:
                            gradients[self.fdict[f]] -= p

        self.W -= c * self.W
        self.W += gradients

    def forward(self, wordseq):
        T = len(wordseq)
        alpha = np.zeros((T, self.n))

        fvector = self.instantiate(wordseq, 0, self.BOS)
        alpha[0] = self.score(fvector)

        for i in range(1, T):
            fvectors = [self.instantiate(wordseq, i, prev_tag)
                        for prev_tag in self.tags]
            scores = np.column_stack([self.score(fv) for fv in fvectors])
            alpha[i] = logsumexp(scores + alpha[i - 1], axis=1)
        return alpha

    def backward(self, wordseq):
        T = len(wordseq)
        beta = np.zeros((T, self.n))

        for i in reversed(range(T - 1)):
            fvectors = [self.instantiate(wordseq, i + 1, prev_tag)
                        for prev_tag in self.tags]
            scores = np.array([self.score(fv) for fv in fvectors])
            beta[i] = logsumexp(scores + beta[i + 1], axis=1)
        return beta

    def predict(self, wordseq):
        T = len(wordseq)
        delta = np.zeros((T, self.n))
        paths = np.zeros((T, self.n), dtype='int')

        fvector = self.instantiate(wordseq, 0, self.BOS)
        delta[0] = self.score(fvector)

        for i in range(1, T):
            fvectors = [self.instantiate(wordseq, i, prev_tag)
                        for prev_tag in self.tags]
            scores = np.array([delta[i - 1][j] + self.score(fv)
                               for j, fv in enumerate(fvectors)])
            paths[i] = np.argmax(scores, axis=0)
            delta[i] = scores[paths[i], np.arange(self.n)]
        prev = np.argmax(delta[-1])

        predict = [prev]
        for i in range(T - 1, 0, -1):
            prev = paths[i, prev]
            predict.append(prev)
        return [self.tags[i] for i in reversed(predict)]

    def score(self, fvector):
        scores = [self.W[self.fdict[f]]
                  for f in fvector if f in self.fdict]
        return np.sum(scores, axis=0)

    def instantiate(self, wordseq, index, prev_tag):
        word = wordseq[index]
        prev_word = wordseq[index - 1] if index > 0 else "^^"
        next_word = wordseq[index + 1] if index < len(wordseq) - 1 else "$$"
        prev_char = prev_word[-1]
        next_char = next_word[0]
        first_char = word[0]
        last_char = word[-1]

        fvector = []
        fvector.append(('01', prev_tag))
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

    def evaluate(self, sentences):
        tp, total = 0, 0

        for sentence in sentences:
            total += len(sentence)
            wordseq, tagseq = zip(*sentence)
            preseq = self.predict(wordseq)
            tp += sum([t == p for t, p in zip(tagseq, preseq)])
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
