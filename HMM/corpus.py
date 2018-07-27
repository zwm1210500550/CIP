# -*- coding: utf-8 -*-

import numpy as np


class Corpus(object):
    UNK = '<UNK>'

    def __init__(self, fdata):
        self.sentences = self.preprocess(fdata)
        self.wordseqs, self.tagseqs = zip(*self.sentences)
        self.words = sorted(set(np.hstack(self.wordseqs)))
        self.tags = sorted(set(np.hstack(self.tagseqs)))
        self.words.append(self.UNK)

        self.wdict = {w: i for i, w in enumerate(self.words)}
        self.tdict = {t: i for i, t in enumerate(self.tags)}
        self.ui = self.wdict[self.UNK]
        self.nw = len(self.words)
        self.nt = len(self.tags)

    def preprocess(self, fdata):
        start = 0
        sentences = []
        with open(fdata, 'r') as train:
            lines = [line for line in train]
        for i, line in enumerate(lines):
            if len(lines[i]) <= 1:
                splits = [l.split()[1:4:2] for l in lines[start:i]]
                wordseq, tagseq = zip(*splits)
                start = i + 1
                while start < len(lines) and len(lines[start]) <= 1:
                    start += 1
                sentences.append((wordseq, tagseq))
        return sentences

    def load(self, fdata):
        data = []
        sentences = self.preprocess(fdata)

        for wordseq, tagseq in sentences:
            wis = [self.wdict[w] if w in self.wdict else self.ui
                   for w in wordseq]
            eis = [self.tdict[t] for t in tagseq]
            data.append((wis, eis))
        return data

    def size(self):
        return self.nw - 1, self.nt
