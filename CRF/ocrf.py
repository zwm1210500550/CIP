# -*- coding: utf-8 -*-

import pickle
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
        # 特征空间
        self.epsilon = list({
            f for wordseq, tagseq in sentences
            for f in set(
                self.instantiate(wordseq, 0, self.BOS)
            ).union(*[
                self.instantiate(wordseq, i, tagseq[i - 1])
                for i, tag in enumerate(tagseq[1:], 1)
            ])
        })
        # 特征对应索引的字典
        self.fdict = {f: i for i, f in enumerate(self.epsilon)}
        # 特征空间维度
        self.d = len(self.epsilon)

        # 特征权重
        self.W = np.zeros((self.d, self.n))

    def SGD(self, train, dev, file,
            epochs, batch_size, c, eta, interval,
            shuffle):
        max_e, max_precision = 0, 0.0
        # 迭代指定次数训练模型
        for epoch in range(epochs):
            start = time.time()
            # 随机打乱数据
            if shuffle:
                random.shuffle(train)
            # 按照指定大小对数据分割批次
            batches = [train[i:i + batch_size]
                       for i in range(0, len(train), batch_size)]
            # 根据批次数据更新权重
            for batch in batches:
                self.update(batch, c, max(eta, 0.00001))
                # eta *= 0.999

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

        for wordseq, tagseq in batch:
            prev_tag = self.BOS
            for i, tag in enumerate(tagseq):
                for f in self.instantiate(wordseq, i, prev_tag):
                    if f in self.fdict:
                        gradients[self.fdict[f], self.tdict[tag]] += 1
                prev_tag = tag

            alpha = self.forward(wordseq)
            beta = self.backward(wordseq)
            logZ = logsumexp(alpha[-1])

            fv = self.instantiate(wordseq, 0, self.BOS)
            p = np.exp(self.score(fv) + beta[0] - logZ)
            for f in fv:
                if f in self.fdict:
                    gradients[self.fdict[f]] -= p

            for i in range(1, len(tagseq)):
                fvs = [self.instantiate(wordseq, i, prev_tag)
                       for prev_tag in self.tags]
                for j, fv in enumerate(fvs):
                    score = self.score(fv)
                    p = np.exp(score + alpha[i - 1, j] + beta[i] - logZ)
                    for f in fv:
                        if f in self.fdict:
                            gradients[self.fdict[f]] -= p

        self.W *= (1 - c)
        for fi, v in gradients.items():
            self.W[fi] += eta * v

    def forward(self, wordseq):
        T = len(wordseq)
        alpha = np.zeros((T, self.n))

        fv = self.instantiate(wordseq, 0, self.BOS)
        alpha[0] = self.score(fv)

        for i in range(1, T):
            fvs = [self.instantiate(wordseq, i, prev_tag)
                   for prev_tag in self.tags]
            scores = np.column_stack([self.score(fv) for fv in fvs])
            alpha[i] = logsumexp(scores + alpha[i - 1], axis=1)
        return alpha

    def backward(self, wordseq):
        T = len(wordseq)
        beta = np.zeros((T, self.n))

        for i in reversed(range(T - 1)):
            fvs = [self.instantiate(wordseq, i + 1, prev_tag)
                   for prev_tag in self.tags]
            scores = np.array([self.score(fv) for fv in fvs])
            beta[i] = logsumexp(scores + beta[i + 1], axis=1)
        return beta

    def predict(self, wordseq):
        T = len(wordseq)
        delta = np.zeros((T, self.n))
        paths = np.zeros((T, self.n), dtype='int')

        fv = self.instantiate(wordseq, 0, self.BOS)
        delta[0] = self.score(fv)

        for i in range(1, T):
            fvs = [self.instantiate(wordseq, i, prev_tag)
                   for prev_tag in self.tags]
            scores = np.array([delta[i - 1][j] + self.score(fv)
                               for j, fv in enumerate(fvs)])
            paths[i] = np.argmax(scores, axis=0)
            delta[i] = scores[paths[i], np.arange(self.n)]
        prev = np.argmax(delta[-1])

        predict = [prev]
        for i in range(T - 1, 0, -1):
            prev = paths[i, prev]
            predict.append(prev)
        return [self.tags[i] for i in reversed(predict)]

    def score(self, fvector):
        scores = np.array([self.W[self.fdict[f]]
                           for f in fvector if f in self.fdict])
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

        for wordseq, tagseq in sentences:
            total += len(wordseq)
            preseq = np.array(self.predict(wordseq))
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
