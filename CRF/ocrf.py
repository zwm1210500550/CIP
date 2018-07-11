# -*- coding: utf-8 -*-

import pickle
from collections import defaultdict
from datetime import datetime, timedelta

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
            epochs, batch_size, interval, c, eta, decay,
            anneal, regularize, shuffle):
        # 记录更新次数
        count = 0
        # 记录迭代时间
        total_time = timedelta()
        # 记录最大准确率及对应的迭代次数
        max_e, max_precision = 0, 0.0
        # 迭代指定次数训练模型
        for epoch in range(epochs):
            start = datetime.now()
            # 随机打乱数据
            if shuffle:
                random.shuffle(train)
            # 设置L2正则化系数
            if not regularize:
                c = 0
            # 按照指定大小对数据分割批次
            batches = [train[i:i + batch_size]
                       for i in range(0, len(train), batch_size)]
            length = len(batches)
            # 根据批次数据更新权重
            for batch in batches:
                if not anneal:
                    self.update(batch, c)
                # 设置学习速率的指数衰减
                else:
                    self.update(batch, c, eta * decay ** (count / length))
                count += 1

            print("Epoch %d / %d: " % (epoch, epochs))
            print("\ttrain: %d / %d = %4f" % self.evaluate(train))
            tp, total, precision = self.evaluate(dev)
            print("\tdev: %d / %d = %4f" % (tp, total, precision))
            t = datetime.now() - start
            print("\t%ss elapsed" % t)
            total_time += t

            # 保存效果最好的模型
            if precision > max_precision:
                self.dump(file)
                max_e, max_precision = epoch, precision
            elif epoch - max_e > interval:
                break
        print("max precision of dev is %4f at epoch %d" %
              (max_precision, max_e))
        print("mean time of each epoch is %s" % (total_time / epoch))

    def update(self, batch, c, eta=1):
        gradients = defaultdict(float)
        bifvs = [self.bigram(prev_tag) for prev_tag in self.tags]
        biscores = np.array([self.score(bifv) for bifv in bifvs])

        for wordseq, tagseq in batch:
            prev_tag = self.BOS
            for i, tag in enumerate(tagseq):
                ti = self.tdict[tag]
                fis = (self.fdict[f]
                       for f in self.instantiate(wordseq, i, prev_tag)
                       if f in self.fdict)
                for fi in fis:
                    gradients[fi, ti] += 1
                prev_tag = tag

            alpha = self.forward(wordseq)
            beta = self.backward(wordseq)
            logZ = logsumexp(alpha[-1])

            fv = self.instantiate(wordseq, 0, self.BOS)
            fis = (self.fdict[f] for f in fv if f in self.fdict)
            p = np.exp(self.score(fv) + beta[0] - logZ)
            for fi in fis:
                gradients[fi] -= p

            for i in range(1, len(tagseq)):
                unifv = self.unigram(wordseq, i)
                unifis = [self.fdict[f] for f in unifv if f in self.fdict]
                scores = biscores + self.score(unifv)
                probs = np.exp(scores + alpha[i - 1][:, None] + beta[i] - logZ)

                for bifv, p in zip(bifvs, probs):
                    bifis = [self.fdict[f] for f in bifv if f in self.fdict]
                    for fi in bifis + unifis:
                        gradients[fi] -= p

        if c != 0:
            self.W *= (1 - eta * c)
        for k, v in gradients.items():
            self.W[k] += eta * v

    def forward(self, wordseq):
        T = len(wordseq)
        alpha = np.zeros((T, self.n))
        bifvs = [self.bigram(prev_tag) for prev_tag in self.tags]
        biscores = np.array([self.score(bifv) for bifv in bifvs])

        fv = self.instantiate(wordseq, 0, self.BOS)
        alpha[0] = self.score(fv)

        for i in range(1, T):
            uniscores = self.score(self.unigram(wordseq, i))
            scores = np.transpose(biscores + uniscores)
            alpha[i] = logsumexp(scores + alpha[i - 1], axis=1)
        return alpha

    def backward(self, wordseq):
        T = len(wordseq)
        beta = np.zeros((T, self.n))
        bifvs = [self.bigram(prev_tag) for prev_tag in self.tags]
        biscores = np.array([self.score(bifv) for bifv in bifvs])

        for i in reversed(range(T - 1)):
            uniscores = self.score(self.unigram(wordseq, i + 1))
            scores = biscores + uniscores
            beta[i] = logsumexp(scores + beta[i + 1], axis=1)
        return beta

    def predict(self, wordseq):
        T = len(wordseq)
        delta = np.zeros((T, self.n))
        paths = np.zeros((T, self.n), dtype='int')
        bifvs = [self.bigram(prev_tag) for prev_tag in self.tags]
        biscores = np.array([self.score(bifv) for bifv in bifvs])

        fv = self.instantiate(wordseq, 0, self.BOS)
        delta[0] = self.score(fv)

        for i in range(1, T):
            uniscores = self.score(self.unigram(wordseq, i))
            scores = np.transpose(biscores + uniscores) + delta[i - 1]
            paths[i] = np.argmax(scores, axis=1)
            delta[i] = scores[np.arange(self.n), paths[i]]
        prev = np.argmax(delta[-1])

        predict = [prev]
        for i in reversed(range(1, T)):
            prev = paths[i, prev]
            predict.append(prev)
        return [self.tags[i] for i in reversed(predict)]

    def score(self, fvector):
        scores = np.array([self.W[self.fdict[f]]
                           for f in fvector if f in self.fdict])
        return np.sum(scores, axis=0)

    def bigram(self, prev_tag):
        return [('01', prev_tag)]

    def unigram(self, wordseq, index):
        word = wordseq[index]
        prev_word = wordseq[index - 1] if index > 0 else '^^'
        next_word = wordseq[index + 1] if index < len(wordseq) - 1 else '$$'
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

    def instantiate(self, wordseq, index, prev_tag):
        bigram = self.bigram(prev_tag)
        unigram = self.unigram(wordseq, index)
        return bigram + unigram

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
            crf = pickle.load(f)
        return crf
