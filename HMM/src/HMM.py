#!/usr/bin/python3
# -*- coding:utf8 -*-

import numpy as np 
import datetime
from config import config

train_data_file = config['train_data_file']
test_data_file = config['test_data_file']
alpha = config['alpha']


def data_handle(data):  
    sentence = list()
    sentences = list()  
    with open(data,'r') as dataTxt:
        for line in dataTxt:
            if len(line) > 1:
                word_and_tag = (line.split()[1] , line.split()[3])
                sentence.append(word_and_tag)
            else:
                sentences.append(sentence)
                sentence = []     #注意sentence = [] 和 sentence.clear()的区别
    return sentences     


class HMM(object):
    def __init__(self,sentences):
        words = set()
        tags = set()
        self.sentences = sentences
        for sentence in sentences:
            for word,tag in sentence:
                words.add(word)
                tags.add(tag)
        words_list = list(words)
        tags_list = list(tags)
        tags_list.append("*start")   #加入开始词性
        tags_list.append("*stop")    #加入结束词性
        words_list.append("???")     #加入未知词
        self.words = {word:index for index,word in enumerate(words_list)} 
        self.tags = {tag:index for index,tag in enumerate(tags_list)}
        self.M = len(self.words)     #词的个数,包括未知词
        self.N = len(self.tags)      #词性个数,包括开始词性和结束词性
        self.transport_matrix = np.zeros((self.N-1,self.N-1))     #最后一行表示从开始词性转移到各词性,最后一列表示转移到结束词性
        self.launch_matrix = np.zeros((self.N-2,self.M))            #最后一列表示发射到未知词

    def launch(self,alpha):
        for sentence in self.sentences:
            for word,tag in sentence:
                self.launch_matrix[self.tags.get(tag)][self.words.get(word)] += 1
        for i in range(len(self.launch_matrix)):
            sum_line = sum(self.launch_matrix[i])
            for j in range(len(self.launch_matrix[i])):
                self.launch_matrix[i][j] = (self.launch_matrix[i][j] + alpha) / (sum_line + alpha * self.M)

    def transport(self,alpha):
        for sentence in self.sentences:
            pre = -1
            for word,tag in sentence:
                self.transport_matrix[pre][self.tags.get(tag)] += 1
                pre = self.tags.get(tag)
            self.transport_matrix[pre][-1] += 1
        for i in range(len(self.transport_matrix)):
            sum_line = sum(self.transport_matrix[i])
            for j in range(len(self.transport_matrix[i])):
                self.transport_matrix[i][j] = (self.transport_matrix[i][j] + alpha) / (sum_line + alpha * (self.N-1))

    def viterbi(self, sentence):
        word_index = list()
        for word in sentence:
            if word in self.words:
                word_index.append(self.words.get(word))
            else:
                word_index.append(self.words.get("???"))

        observeNum = len(sentence) + 1              #句子长度加一
        tagNum = self.N - 2                         #词性数
        max_p = np.zeros((observeNum, tagNum))      #第一行用于初始化,max_p[i][j]表示从开始到第i个观测对应第j个词性的概率最大值
        path = np.zeros((observeNum, tagNum),dtype="int")       #第一行用于初始化,path[i][j]表示从开始到第i个观测对应第j个词性概率最大时i-1个观测的词性索引值

        transport_matrix = np.log(self.transport_matrix)    #对数处理后，点乘运算变为加法运算
        launch_matrix = np.log(self.launch_matrix)
 
        for i in range(tagNum):
            path[0][i] = -1
            max_p[0][i] = transport_matrix[-1][i] + launch_matrix[i][word_index[0]]

        for i in range(1, observeNum):
            if i == observeNum - 1:         #i是最后一个观测
                for k in range(tagNum):
                    max_p[i][k] = max_p[i - 1][k] + transport_matrix[k][-1]
                    last_path = k
                    path[i][k] = last_path
            else:
                for j in range(tagNum):
                    prob = max_p[i - 1] + transport_matrix[:-1, j] + launch_matrix[j][word_index[i]]
                    path[i][j] = np.argmax(prob)
                    max_p[i][j] = max(prob)

        gold_path = []
        cur_state = observeNum - 1      #初始处于结束态
        step = np.argmax(max_p[cur_state])
        while (True):
            step = path[cur_state][step]
            if step == -1:
                break
            gold_path.insert(0, step)
            cur_state -= 1
        return gold_path
    

    def evaluate(self, test_data):
        total_words = 0
        correct_words = 0
        sentence_num = 0
        print('正在评估测试集...')
        for sentence in test_data:
            sentence_num += 1
            word_list = []
            tag_list = []
            for word, tag in sentence:
                word_list.append(word)
                tag_list.append(tag)
            predict = self.viterbi(word_list)
            total_words += len(sentence)
            for i in range(len(predict)):
                if predict[i] == self.tags.get(tag_list[i]):
                    correct_words += 1
        print('共%d个句子' % (sentence_num))
        print('共%d个单词，预测正确%d个单词' % (total_words, correct_words))
        print('准确率：%f' % (correct_words / total_words))

if __name__ == "__main__":
    sentences = data_handle(train_data_file)
    hmm = HMM(sentences)
    hmm.launch(alpha)
    hmm.transport(alpha)
    test_data = data_handle(test_data_file)
    startTime = datetime.datetime.now()
    hmm.evaluate(test_data)
    stopTime = datetime.datetime.now()
    print("用时：" + str(stopTime-startTime))




