# -*- coding: utf-8 -*-

import pickle
import random
from datetime import datetime, timedelta

import numpy as np


class LinearModel(object):

    def __init__(self, nt):
        # 词性数量
        self.nt = nt

    def create_feature_space(self, data):
        # 特征空间
        self.epsilon = list({
            f for wordseq, tiseq in data
            for i, ti in enumerate(tiseq)
            for f in self.instantiate(wordseq, i)
        })
        # 特征对应索引的字典
        self.fdict = {f: i for i, f in enumerate(self.epsilon)}
        # 特征空间维度
        self.d = len(self.epsilon)

        # 特征权重
        self.W = np.zeros((self.d, self.nt))
        # 累加特征权重
        self.V = np.zeros((self.d, self.nt))

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
            self.k, self.R = 0, np.zeros((self.d, self.nt), dtype='int')
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
        print("mean time of each epoch is %ss" % (total_time / (epoch + 1)))

    def update(self, batch):
        wordseq, tiseq = batch
        # 根据单词序列的正确词性更新权重
        for i, ti in enumerate(tiseq):
            # 根据现有权重向量预测词性
            pi = self.predict(wordseq, i)
            # 如果预测词性与正确词性不同，则更新权重
            if ti != pi:
                fv = self.instantiate(wordseq, i)
                fiseq = (self.fdict[f] for f in fv if f in self.fdict)
                for fi in fiseq:
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
        return np.argmax(scores)

    def score(self, fvector, average=False):
        # 获取特征索引
        fiseq = [self.fdict[f] for f in fvector if f in self.fdict]
        # 计算特征对应权重得分
        scores = self.V[fiseq] if average else self.W[fiseq]
        return np.sum(scores, axis=0)

    def instantiate(self, wordseq, index):
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

    def evaluate(self, data, average=False):
        tp, total = 0, 0

        for wordseq, tiseq in data:
            total += len(wordseq)
            piseq = np.array([self.predict(wordseq, i, average)
                              for i in range(len(wordseq))])
            tp += np.sum(tiseq == piseq)
        precision = tp / total
        return tp, total, precision

    def dump(self, file):
        with open(file, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, file):
        with open(file, 'rb') as f:
            lm = pickle.load(f)
        return lm
