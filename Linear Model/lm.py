# -*- coding: utf-8 -*-

import time

import numpy as np


def preprocessing(ftrain):
    with open(ftrain, 'r') as train:
        it = iter(train)
        sentences = []
        for line in it:
            sentence = []
            while len(line) > 1:
                splits = line.split()
                sentence.append((splits[1], splits[3]))
                line = next(it)
            sentences.append(sentence)
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
                features = self.instantialize(wordseq, i, tag)
                feature_space.update(features)

        # 特征空间
        self.epsilon = list(feature_space)
        # 特征对应索引的字典
        self.feadict = {f: i for i, f in enumerate(self.epsilon)}
        # 特征空间维度
        self.D = len(self.epsilon)

        # 特征权重
        self.weights = np.zeros(self.D, dtype='int')
        # 平均特征权重
        self.average_weights = np.zeros(self.D, dtype='int')

    def online_train(self, sentences, iter=20):
        for it in range(iter):
            for sentence in sentences:
                wordseq, tagseq = zip(*sentence)
                # 根据单词序列的正确词性更新权重
                for i, tag in enumerate(tagseq):
                    self.update(wordseq, i, tag)
            yield it

    def update(self, wordseq, index, tag):
        # 根据现有权重向量预测词性
        pre = self.predict(wordseq, index)
        # 如果预测词性与正确词性不同，则更新权重
        if tag != pre:
            cor_features = self.instantialize(wordseq, index, tag)
            err_features = self.instantialize(wordseq, index, pre)
            for cf, ef in zip(cor_features, err_features):
                if cf in self.feadict:
                    self.weights[self.feadict[cf]] += 1
                if ef in self.feadict:
                    self.weights[self.feadict[ef]] -= 1
            self.average_weights += self.weights

    def predict(self, wordseq, index, average=False):
        i = np.argmax([
            self.score(self.instantialize(wordseq, index, tag),
                       average=average)
            for tag in self.tags
        ])
        return self.tags[i]

    def score(self, features, average=False):
        # 计算特征对应累加权重的得分
        if average:
            scores = [self.average_weights[self.feadict[f]]
                      for f in features if f in self.feadict]
        # 计算特征对应未累加权重的得分
        else:
            scores = [self.weights[self.feadict[f]]
                      for f in features if f in self.feadict]
        return np.sum(scores)

    def instantialize(self, wordseq, index, tag):
        word = wordseq[index]
        prev_word = wordseq[index - 1] if index > 0 else "$$"
        next_word = wordseq[index + 1] if index < len(wordseq) - 1 else "##"
        prev_char = prev_word[-1]
        next_char = next_word[0]
        first_char = word[0]
        last_char = word[-1]

        features = []
        features.append(('02', tag, word))
        features.append(('03', tag, prev_word))
        features.append(('04', tag, next_word))
        features.append(('05', tag, word, prev_char))
        features.append(('06', tag, word, next_char))
        features.append(('07', tag, first_char))
        features.append(('08', tag, last_char))

        for char in word[1:-1]:
            features.append(('09', tag, char))
            features.append(('10', tag, first_char, char))
            features.append(('11', tag, last_char, char))
        if len(word) == 1:
            features.append(('12', tag, word, prev_char, next_char))
        for i in range(1, len(word)):
            prev_char, char = word[i - 1], word[i]
            if prev_char == char:
                features.append(('13', tag, char, 'consecutive'))
            if i <= 4:
                features.append(('14', tag, word[:i]))
                features.append(('15', tag, word[-i:]))
        if len(word) <= 4:
            features.append(('14', tag, word))
            features.append(('15', tag, word))
        return features

    def evaluate(self, sentences):
        tp, total = 0, 0
        tagseqs, preseqs = [], []

        for sentence in sentences:
            ws, ts = zip(*sentence)
            tagseqs.append(ts)
            preseqs.append([lm.predict(ws, i, False) for i in range(len(ws))])
        for tagseq, preseq in zip(tagseqs, preseqs):
            total += len(tagseq)
            tp += sum(t == p for t, p in zip(tagseq, preseq))
        precision = tp / total
        return tp, total, precision


if __name__ == '__main__':
    train = preprocessing('data/train.conll')
    dev = preprocessing('data/dev.conll')

    all_words, all_tags = zip(*np.vstack(train))
    tags = sorted(list(set(all_tags)))

    start = time.time()

    print("Creating Linear Model with %d tags" % (len(tags)))
    lm = LinearModel(tags)

    print("Using %d sentences to create the feature space" % (len(train)))
    lm.create_feature_space(train)
    print("The size of the feature space is %d" % lm.D)

    evaluations = []

    print("Using online-training algorithm to train the model")
    for i in lm.online_train(train):
        print("iteration %d" % i)
        result = lm.evaluate(train)
        print("\ttrain: %d / %d = %4f" % result)
        result = lm.evaluate(dev)
        print("\tdev: %d / %d = %4f" % result)
        evaluations.append(result)

    print("Successfully evaluated dev data using the model")
    print("precision: %d / %d = %4f" % max(evaluations, key=lambda x: x[2]))
    print("%4fs elapsed" % (time.time() - start))
