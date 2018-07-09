# -*- coding: utf-8 -*-


class Config(object):

    def __init__(self, bigdata=False):
        self.epochs = 100
        self.batch_size = 50
        self.eta = 0.5
        self.c = 0.0001
        if bigdata:
            self.decay = 0.99
            self.interval = 10
            self.ftrain = 'bigdata/train.conll'
            self.fdev = 'bigdata/dev.conll'
            self.ftest = 'bigdata/test.conll'
        else:
            self.decay = 0.96
            self.interval = 5
            self.ftrain = 'data/train.conll'
            self.fdev = 'data/dev.conll'
        self.llmpkl = 'llm.pkl'
