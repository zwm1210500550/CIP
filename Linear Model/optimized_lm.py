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

    def __init__(self, words, tags):
        # 所有不同的单词
        self.words = words
        # 所有不同的词性
        self.tags = tags
        # 单词对应索引的字典
        self.wordict = {w: i for i, w in enumerate(words)}
        # 词性对应索引的字典
        self.tagdict = {t: i for i, t in enumerate(tags)}

        self.M = len(self.words)
        self.N = len(self.tags)

    def create_feature_space(self, sentences):
        feature_space = set()
        for sentence in sentences:
            wordseq, tagseq = zip(*sentence)
            for i, tag in enumerate(tagseq):
                features = self.instantialize(wordseq, i)
                feature_space.update(features)

        # 特征空间
        self.epsilon = list(feature_space)
        # 特征对应索引的字典
        self.feadict = {f: i for i, f in enumerate(self.epsilon)}
        # 特征空间维度
        self.D = len(self.epsilon)

        # 特征权重
        self.weights = np.zeros((self.D, self.N), dtype='int')

    def online_train(self, sentences, iter=20):
        for it in range(iter):
            for sentence in sentences:
                wordseq, tagseq = zip(*sentence)
                # 根据单词序列的正确词性更新权重
                for i, tag in enumerate(tagseq):
                    self.update(wordseq, i, tag)

            tagseqs, preseqs = [], []
            for sentence in sentences:
                ws, ts = zip(*sentence)
                tagseqs.append(ts)
                preseqs.append([lm.predict(ws, i) for i in range(len(ws))])
            tp, total, precision = lm.evaluate(tagseqs, preseqs)
            print('iteration %d: %d / %d = %4f' % (it, tp, total, precision))

    def instantialize(self, wordseq, index):
        word = wordseq[index]
        prev_word = wordseq[index - 1] if index > 0 else "$$"
        next_word = wordseq[index + 1] if index < len(wordseq) - 1 else "##"
        prev_char = prev_word[-1]
        next_char = next_word[0]
        first_char = word[0]
        last_char = word[-1]

        features = []
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

    def update(self, wordseq, index, tag):
        # 根据现有权重向量预测词性
        pre = self.predict(wordseq, index)
        # 如果预测词性与正确词性不同，则更新权重
        if tag != pre:
            for feature in self.instantialize(wordseq, index):
                if feature in self.feadict:
                    f_index = self.feadict[feature]
                    t_index = self.tagdict[tag]
                    p_index = self.tagdict[pre]
                    self.weights[f_index][t_index] += 1
                    self.weights[f_index][p_index] -= 1

    def predict(self, wordseq, index):
        features = self.instantialize(wordseq, index)
        i = np.argmax(self.score(features))
        return self.tags[i]

    def score(self, features):
        return np.sum([self.weights[self.feadict[f]]
                       for f in features if f in self.feadict],
                      axis=0)

    def evaluate(self, tagseqs, predicts):
        tp, total = 0, 0
        for tagseq, predict in zip(tagseqs, predicts):
            total += len(tagseq)
            tp += sum(t == p for t, p in zip(tagseq, predict))
        precision = tp / total
        return tp, total, precision


if __name__ == '__main__':
    sentences = preprocessing('data/train.conll')

    all_words, all_tags = zip(*np.vstack(sentences))
    words, tags = list(set(all_words)), list(set(all_tags))

    start = time.time()

    print("Creating Linear Model with %d words and %d tags"
          % (len(words), len(tags)))
    lm = LinearModel(words, tags)

    print("Using %d sentences to create the feature space"
          % (len(sentences)))
    lm.create_feature_space(sentences)
    print("The size of the feature space is %d" % lm.D)

    print("Using online-training algorithm to train the model")
    lm.online_train(sentences)

    sentences = preprocessing('data/dev.conll')
    tagseqs, preseqs = [], []

    print("Using the trained model to tag %d sentences"
          % len(sentences))
    for sentence in sentences:
        ws, ts = zip(*sentence)
        tagseqs.append(ts)
        preseqs.append([lm.predict(ws, i) for i in range(len(ws))])

    print("Evaluating the result")
    tp, total, precision = lm.evaluate(tagseqs, preseqs)
    print("precision: %d / %d = %4f" % (tp, total, precision))
    print("%4fs elapsed" % (time.time() - start))
