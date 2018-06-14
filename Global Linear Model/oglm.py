# -*- coding: utf-8 -*-

import sys
import time

import numpy as np


def preprocessing(ftrain):
    start = 0
    sentences = []
    with open(ftrain, 'r') as train:
        lines = [line for line in train]
    for i, line in enumerate(lines):
        if len(lines[i]) <= 1:
            sentences.append([l.split()[1:4:2] for l in lines[start:i]])
            start = i + 1
            while start < len(lines) and len(lines[start]) <= 1:
                start += 1
    return sentences


class GlobalLinearModel(object):

    def __init__(self, tags):
        # 所有不同的词性
        self.tags = tags
        # 句首词性
        self.BOS = 'BOS'
        # 词性对应索引的字典
        self.tagdict = {t: i for i, t in enumerate(tags)}

        self.N = len(self.tags)

    def create_feature_space(self, sentences):
        feature_space = set()
        for sentence in sentences:
            wordseq, tagseq = zip(*sentence)
            prev_tag = self.BOS
            for i, tag in enumerate(tagseq):
                features = self.instantialize(wordseq, i, prev_tag)
                feature_space.update(features)
                prev_tag = tag

        # 特征空间
        self.epsilon = list(feature_space)
        # 特征对应索引的字典
        self.feadict = {f: i for i, f in enumerate(self.epsilon)}
        # 特征空间维度
        self.D = len(self.epsilon)

        # 特征权重
        self.W = np.zeros((self.D, self.N), dtype='int')
        # 累加特征权重
        self.V = np.zeros((self.D, self.N), dtype='int')

    def online(self, sentences, epochs=20):
        for epoch in range(epochs):
            for sentence in sentences:
                wordseq, tagseq = zip(*sentence)
                # 根据单词序列的正确词性更新权重
                self.update(wordseq, tagseq)
            yield epoch

    def update(self, wordseq, tagseq):
        # 根据现有权重向量预测词性序列
        preseq = self.predict(wordseq)
        # 如果预测词性序列与正确词性序列不同，则更新权重
        if tagseq != preseq:

            prev_tag, prev_pre = self.BOS, self.BOS
            for i, (tag, pre) in enumerate(zip(tagseq, preseq)):
                t_index, p_index = self.tagdict[tag], self.tagdict[pre]
                for cf in self.instantialize(wordseq, i, prev_tag):
                    if cf in self.feadict:
                        self.W[self.feadict[cf]][t_index] += 1
                for ef in self.instantialize(wordseq, i, prev_pre):
                    if ef in self.feadict:
                        self.W[self.feadict[ef]][p_index] -= 1
                prev_tag, prev_pre = tag, pre
            # self.V += self.W

    def predict(self, wordseq, average=False):
        T = len(wordseq)
        delta = np.zeros((T, self.N))
        paths = np.zeros((T, self.N), dtype='int')

        delta[0] = self.score(self.instantialize(wordseq, 0, self.BOS))

        for i in range(1, T):
            tag_features = [
                self.instantialize(wordseq, i, prev_tag)
                for prev_tag in self.tags
            ]
            scores = [delta[i - 1][j] + self.score(fs)
                      for j, fs in enumerate(tag_features)]
            paths[i] = np.argmax(scores, axis=0)
            delta[i] = np.max(scores, axis=0)
        prev = np.argmax(delta[-1])

        predict = [prev]
        for i in range(T - 1, 0, -1):
            prev = paths[i, prev]
            predict.append(prev)
        return [self.tags[i] for i in reversed(predict)]

    def score(self, features, average=False):
        # 计算特征对应累加权重的得分
        if average:
            scores = [self.V[self.feadict[f]]
                      for f in features if f in self.feadict]
        # 计算特征对应未累加权重的得分
        else:
            scores = [self.W[self.feadict[f]]
                      for f in features if f in self.feadict]
        return np.sum(scores, axis=0)

    def instantialize(self, wordseq, index, prev_tag):
        word = wordseq[index]
        prev_word = wordseq[index - 1] if index > 0 else "$$"
        next_word = wordseq[index + 1] if index < len(wordseq) - 1 else "##"
        prev_char = prev_word[-1]
        next_char = next_word[0]
        first_char = word[0]
        last_char = word[-1]

        features = []
        features.append(('01', prev_tag))
        features.append(('02', word))
        features.append(('03', prev_word))
        features.append(('04', next_word))
        features.append(('05', word, prev_char))
        features.append(('06', word, next_char))
        features.append(('07', first_char))
        features.append(('08', last_char))

        for char in word[1:-1]:
            features.append(('09', char))
            features.append(('10', first_char, char))
            features.append(('11', last_char, char))
        if len(word) == 1:
            features.append(('12', word, prev_char, next_char))
        for i in range(1, len(word)):
            prev_char, char = word[i - 1], word[i]
            if prev_char == char:
                features.append(('13', char, 'consecutive'))
            if i <= 4:
                features.append(('14', word[:i]))
                features.append(('15', word[-i:]))
        if len(word) <= 4:
            features.append(('14', word))
            features.append(('15', word))
        return features

    def evaluate(self, sentences, average=False):
        tp, total = 0, 0

        for sentence in sentences:
            total += len(sentence)
            wordseq, tagseq = zip(*sentence)
            preseq = self.predict(wordseq, average)
            tp += sum([t == p for t, p in zip(tagseq, preseq)])
        precision = tp / total
        return tp, total, precision


if __name__ == '__main__':
    train = preprocessing('data/train.conll')
    dev = preprocessing('data/dev.conll')

    all_words, all_tags = zip(*np.vstack(train))
    tags = sorted(set(all_tags))

    start = time.time()

    print("Creating Linear Model with %d tags" % (len(tags)))
    glm = GlobalLinearModel(tags)

    print("Using %d sentences to create the feature space" % (len(train)))
    glm.create_feature_space(train)
    print("The size of the feature space is %d" % glm.D)

    average = len(sys.argv) > 1 and sys.argv[1] == 'average'
    evaluations = []

    print("Using online-training algorithm to train the model")
    for epoch in glm.online(train):
        print("Epoch %d" % epoch)
        result = glm.evaluate(train, average=average)
        print("\ttrain: %d / %d = %4f" % result)
        result = glm.evaluate(dev, average=average)
        print("\tdev: %d / %d = %4f" % result)
        evaluations.append(result)

    print("Successfully evaluated dev data using the model")
    print("Precision: %d / %d = %4f" % max(evaluations, key=lambda x: x[2]))
    print("%4fs elapsed" % (time.time() - start))
