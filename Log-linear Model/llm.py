# -*- coding: utf-8 -*-

import time
import random

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


class LogLinearModel(object):

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
        self.W = np.zeros(self.D)

    def SGD(self, sentences, B=50, C=0.0001, eta=0.5, iter=10):
        training_data = []
        for sentence in sentences:
            wordseq, tagseq = zip(*sentence)
            for i, tag in enumerate(tagseq):
                training_data.append((wordseq, i, tag))
        for it in range(iter):
            random.shuffle(training_data)
            batches = [training_data[i:i + B]
                       for i in range(0, len(training_data), B)]
            for batch in batches:
                # 根据批次数据更新权重
                self.update(batch, C, max(eta, 0.00001))
                eta *= 0.999
            yield it

    def update(self, batch, C, eta):
        gradients = np.zeros(self.D)
        for wordseq, i, tag in batch:
            for cf in self.instantialize(wordseq, i, tag):
                if cf in self.feadict:
                    gradients[self.feadict[cf]] += 1
            # 获取每个词性对应的所有特征
            tag_features = [self.instantialize(wordseq, i, t)
                            for t in self.tags]
            scores = [self.score(features) for features in tag_features]
            probs = np.exp(scores) / sum(np.exp(scores))

            for features, p in zip(tag_features, probs):
                for f in features:
                    if f in self.feadict:
                        gradients[self.feadict[f]] -= p
        self.W -= eta * C * self.W
        self.W += eta * gradients

    def predict(self, wordseq, index):
        tag_features = [self.instantialize(wordseq, index, tag)
                        for tag in self.tags]
        scores = [self.score(features)
                  for features in tag_features]
        return self.tags[np.argmax(scores)]

    def score(self, features):
        scores = [self.W[self.feadict[f]]
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

        for sentence in sentences:
            total += len(sentence)
            wordseq, tagseq = zip(*sentence)
            preseq = [self.predict(wordseq, i)
                      for i in range(len(wordseq))]
            tp += sum([t == p for t, p in zip(tagseq, preseq)])
        precision = tp / total
        return tp, total, precision


if __name__ == '__main__':
    train = preprocessing('data/train.conll')
    dev = preprocessing('data/dev.conll')

    all_words, all_tags = zip(*np.vstack(train))
    tags = sorted(set(all_tags))

    start = time.time()

    print("Creating Log-Linear Model with %d tags" % (len(tags)))
    llm = LogLinearModel(tags)

    print("Using %d sentences to create the feature space" % (len(train)))
    llm.create_feature_space(train)
    print("The size of the feature space is %d" % llm.D)

    evaluations = []

    print("Using SGD-training algorithm to train the model")
    for i in llm.SGD(train):
        print("iteration %d" % i)
        result = llm.evaluate(train)
        print("\ttrain: %d / %d = %4f" % result)
        result = llm.evaluate(dev)
        print("\tdev: %d / %d = %4f" % result)
        evaluations.append(result)

    print("Successfully evaluated dev data using the model")
    print("Precision: %d / %d = %4f" % max(evaluations, key=lambda x: x[2]))
    print("%4fs elapsed" % (time.time() - start))
