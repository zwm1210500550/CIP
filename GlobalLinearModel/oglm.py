# -*- coding: utf-8 -*-

import pickle
import random
from collections import Counter
from datetime import datetime, timedelta

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


class GlobalLinearModel(object):

    def __init__(self, tags):
        # 所有不同的词性
        self.tags = tags
        # 句首词性
        self.BOS = 'BOS'
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
        # 累加特征权重
        self.V = np.zeros((self.d, self.n))
        self.BF = [self.bigram(prev_tag) for prev_tag in self.tags]

    def online(self, train, dev, file, epochs, interval, average, shuffle):
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

    def update(self, batch):
        wordseq, tagseq = batch
        # 根据现有权重向量预测词性序列
        preseq = self.predict(wordseq)
        # 如果预测词性序列与正确词性序列不同，则更新权重
        if not np.array_equal(tagseq, preseq):
            prev_tag, prev_pre = self.BOS, self.BOS
            for i, (tag, pre) in enumerate(zip(tagseq, preseq)):
                ti, pi = self.tdict[tag], self.tdict[pre]

                cfreqs = Counter(self.instantiate(wordseq, i, prev_tag))
                # 统计正确词性不同权重出现的次数
                cfis, cfcounts = map(list, zip(*[
                    (self.fdict[f], cfreqs[f])
                    for f in cfreqs if f in self.fdict
                ]))
                prev_w, prev_r = self.W[cfis, ti], self.R[cfis, ti]
                # 累加权重加上步长乘以权重
                self.V[cfis, ti] += (self.k - prev_r) * prev_w
                # 更新正确词性对应权重
                self.W[cfis, ti] += cfcounts
                # 更新时间戳记录
                self.R[cfis, ti] = self.k

                efreqs = Counter(self.instantiate(wordseq, i, prev_pre))
                # 统计预测词性不同权重出现的次数
                efis, efcounts = map(list, zip(*[
                    (self.fdict[f], efreqs[f])
                    for f in efreqs if f in self.fdict
                ]))
                prev_w, prev_r = self.W[efis, pi], self.R[efis, pi]
                # 累加权重加上步长乘以权重
                self.V[efis, pi] += (self.k - prev_r) * prev_w
                # 更新预测词性对应权重
                self.W[efis, pi] -= efcounts
                # 更新时间戳记录
                self.R[efis, pi] = self.k

                prev_tag, prev_pre = tag, pre
            self.k += 1

    def predict(self, wordseq, average=False):
        T = len(wordseq)
        delta = np.zeros((T, self.n))
        paths = np.zeros((T, self.n), dtype='int')
        biscores = np.array([self.score(bifv, average)
                             for bifv in self.BF])

        fv = self.instantiate(wordseq, 0, self.BOS)
        delta[0] = self.score(fv, average)

        for i in range(1, T):
            uniscores = self.score(self.unigram(wordseq, i), average)
            scores = (biscores + uniscores).T + delta[i - 1]
            paths[i] = np.argmax(scores, axis=1)
            delta[i] = scores[np.arange(self.n), paths[i]]
        prev = np.argmax(delta[-1])

        predict = [prev]
        for i in reversed(range(1, T)):
            prev = paths[i, prev]
            predict.append(prev)
        return [self.tags[i] for i in reversed(predict)]

    def score(self, fvector, average=False):
        # 获取特征索引
        fis = [self.fdict[f] for f in fvector if f in self.fdict]
        # 计算特征对应权重得分
        scores = self.V[fis] if average else self.W[fis]
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

    def evaluate(self, sentences, average=False):
        tp, total = 0, 0

        for wordseq, tagseq in sentences:
            total += len(wordseq)
            preseq = np.array(self.predict(wordseq, average))
            tp += np.sum(tagseq == preseq)
        precision = tp / total
        return tp, total, precision

    def dump(self, file):
        with open(file, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, file):
        with open(file, 'rb') as f:
            glm = pickle.load(f)
        return glm
