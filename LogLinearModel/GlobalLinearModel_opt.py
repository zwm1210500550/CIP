#!\usr\bin\python
# coding=UTF-8
import numpy as np
import time


class GlobalLinearModel:
    def __init__(self):
        self.sentences = []  # 句子的列表
        self.pos = []  # words对应的词性
        self.words = []  # 句子分割成词的词组列表的列表
        self.feature = {}  # 特征向量的字典
        self.tags = []  # 词性的集合 n
        self.len_tags=0
        self.len_feature=0
        self.dic_tags={} # 词性的字典
        self.weight = []  # 特征向量的值
        self.v=[]

        self.sentences_dev=[]
        self.pos_dev=[]
        self.words_dev=[]

        self.start_time = time.time()

    def readfile(self,filename):
        with open(filename, 'r') as ft:
            words=[]
            sentences=[]
            pos=[]
            temp_words = []
            temp_sen = ''
            temp_poss = []
            for line in ft:
                if len(line) > 1:
                    temp_word = line.strip().split('\t')[1].decode('utf-8')
                    temp_pos = line.strip().split('\t')[3]
                    if temp_pos in self.tags:
                        self.tags.append(temp_pos)
                    temp_sen += temp_word
                    temp_words.append(temp_word)
                    temp_poss.append(temp_pos)
                else:
                    words.append(temp_words)
                    sentences.append(temp_sen)
                    pos.append(temp_poss)
                    temp_words = []
                    temp_sen = ''
                    temp_poss = []
            return words,sentences,pos

    def readdata(self):
        self.words,self.sentences,self.pos=self.readfile('train.conll')
        self.words_dev, self.sentences_dev, self.pos_dev = self.readfile('dev.conll')

    def create_feature_templates_global(self, words, index, pre_tag):
        f = []
        # words_of_sentence
        wos = words
        if index == 0:
            wi_minus = '$$'
            ci_minus_minus1 = '$'
            pre_tag = '<BOS>'
        else:
            wi_minus = wos[index - 1]
            ci_minus_minus1 = wi_minus[-1]
        if index == len(wos) - 1:
            wi_plus = '##'
            ci_plus0 = '#'
        else:
            wi_plus = wos[index + 1]
            ci_plus0 = wi_plus[0]
        ci0 = wos[index][0]
        ci_minus1 = wos[index][-1]
        f.append('01:' + str(pre_tag))
        f.append('02:' + wos[index])
        f.append('03:' + wi_minus)
        f.append('04:' + wi_plus)
        f.append('05:' + wos[index] + '*' + ci_minus_minus1)
        f.append('06:' + wos[index] + ci_plus0)
        f.append('07:' + ci0)
        f.append('08:' + ci_minus1)
        for i in range(1, len(wos[index]) - 1):
            f.append('09:' + wos[index][i])
            f.append('10:' + wos[index][0] + '*' + wos[index][i])
            f.append('11:' + wos[index][-1] + '*' + wos[index][i])
            if wos[index][i] == wos[index][i + 1]:
                f.append('13:' + wos[index][i] + '*' + 'consecutive')
        if len(wos[index]) == 1:
            f.append('12:' + ci_minus_minus1 + '*' + ci_plus0)
        for i in range(1, len(wos[index]) + 1):
            if i > 4:
                break
            f.append('14:' + wos[index][0:i+1])
            f.append('15:' + wos[index][-i - 1::])
        return f

    def get_max_score(self, sentence):
        score = 0
        temp_score = 0
        MIN = -10
        path_max = []
        dim_tags = self.len_tags  # 词性维度
        len_sen = len(sentence) # 句子单词个数
        score_prob = np.zeros((len_sen, dim_tags))  # 得分矩阵
        score_path = np.zeros((len_sen, dim_tags), dtype=np.int)  # 路径矩阵
        score_path[0] = np.array([-1] * dim_tags)
        for t in self.dic_tags:
            i = self.dic_tags[t]
            f = self.create_feature_templates_global(sentence, 0, '<BOS>')
            for j in f:
                if j in self.feature:
                    a=self.feature[j]
                    temp_score += self.weight[a+self.len_feature*i]
            score_prob[0][i] = temp_score
            temp_score = 0
        if len_sen > 1:
            for i in range(1, len_sen):
                for t in self.dic_tags:
                    j=self.dic_tags[t]
                    score_prob[i][j] = MIN
                    for k in range(0, dim_tags):  # 路径搜索
                        f = self.create_feature_templates_global(sentence, i, self.tags[k])
                        for m in f:
                            if m in self.feature:
                                temp_score += self.weight[self.feature[m]+self.len_feature*j]
                        temp_prob = score_prob[i - 1][k] + temp_score
                        if temp_prob > score_prob[i][j]:
                            score_prob[i][j] = temp_prob
                            score_path[i][j] = k
                        temp_score = 0
        a = score_prob[len_sen - 1]
        max_point = np.where(a == max(a))[0][0]
        for i in range(len_sen - 1, 0, -1):
            path_max.insert(0, self.tags[max_point])
            max_point = score_path[i][max_point]
        path_max.insert(0, self.tags[max_point])
        return path_max

    def get_max_score_average(self, sentence):
        score = 0
        temp_score = 0
        MIN = -10
        path_max = []
        dim_tags = self.len_tags  # 词性维度
        len_sen = len(sentence) # 句子单词个数
        score_prob = np.zeros((len_sen, dim_tags))  # 得分矩阵
        score_path = np.zeros((len_sen, dim_tags), dtype=np.int)  # 路径矩阵
        score_path[0] = np.array([-1] * dim_tags)
        for t in self.dic_tags:
            i = self.dic_tags[t]
            f = self.create_feature_templates_global(sentence, 0, '<BOS>')
            for j in f:
                if j in self.feature:
                    a=self.feature[j]
                    temp_score += self.v[a+self.len_feature*i]
            score_prob[0][i] = temp_score
            temp_score = 0
        if len_sen > 1:
            for i in range(1, len_sen):
                for t in self.dic_tags:
                    j=self.dic_tags[t]
                    score_prob[i][j] = MIN
                    for k in range(0, dim_tags):  # 路径搜索
                        f = self.create_feature_templates_global(sentence, i, self.tags[k])
                        for m in f:
                            if m in self.feature:
                                temp_score += self.v[self.feature[m]+self.len_feature*j]
                        temp_prob = score_prob[i - 1][k] + temp_score
                        if temp_prob > score_prob[i][j]:
                            score_prob[i][j] = temp_prob
                            score_path[i][j] = k
                        temp_score = 0
        a = score_prob[len_sen - 1]
        max_point = np.where(a == max(a))[0][0]
        for i in range(len_sen - 1, 0, -1):
            path_max.insert(0, self.tags[max_point])
            max_point = score_path[i][max_point]
        path_max.insert(0, self.tags[max_point])
        return path_max

    def create_feature_space(self):
        for index_sen in range(0, len(self.sentences)):
            sen = self.words[index_sen]
            for index_word in range(0, len(sen)):
                tag = self.pos[index_sen][index_word]
                if index_word == 0:
                    pretag = 'NULL'
                else:
                    pretag = self.pos[index_sen][index_word - 1]
                f = self.create_feature_templates_global(sen, index_word, pretag)
                for i in f:
                    if i not in self.feature:
                        self.feature[i] = len(self.feature)
                if tag not in self.tags:
                    self.tags.append(tag)
        self.dic_tags={value:i for i,value in enumerate(self.tags)}
        self.len_feature=len(self.feature)
        self.len_tags=len(self.tags)
        self.weight = np.zeros(self.len_feature * self.len_tags,dtype=np.int)
        self.v = np.zeros(self.len_feature * self.len_tags)
        print "特征向量数目：" + str(self.len_feature)
        print "词性数目：" + str(self.len_tags)

    def perceptron_online_training(self, iteration=20):
        for it in range(0, iteration):
            self.start_time = time.time()
            for index_sen in range(0, len(self.sentences)):
                sen = self.words[index_sen]
                max_tag = self.get_max_score(sen)
                right_tag = self.pos[index_sen]
                if max_tag != right_tag:
                    for i in range(0, len(max_tag)):
                        index_tag_m=self.dic_tags[max_tag[i]]
                        index_tag_p=self.dic_tags[right_tag[i]]
                        if i == 0:
                            pretag_m = 'NULL'
                            pretag_p = 'NULL'
                        else:
                            pretag_m = max_tag[i - 1]
                            pretag_p = right_tag[i - 1]
                        f_m = self.create_feature_templates_global(sen, i, pretag_m)
                        f_p = self.create_feature_templates_global(sen, i, pretag_p)
                        for i_m in f_m:
                            if i_m in self.feature:
                                self.weight[self.feature[i_m]+self.len_feature*index_tag_m] -= 1
                        for i_p in f_p:
                            if i_p in self.feature:
                                self.weight[self.feature[i_p]+self.len_feature*index_tag_p] += 1
                    self.v += self.weight
            #self.testdata('train')
            self.testdata('dev')
            over_time=time.time()
            with open('result2_average.txt', 'a') as fr:
                fr.write('迭代：'+str(it) +'\t'+'用时'+str(over_time-self.start_time)+'s'+'\n')


    def output(self):
        with open('model.txt', 'w') as fw:
            for j in range(0,self.len_tags):
                for i in self.feature:
                    f_index = self.feature[i]
                    value = self.weight[f_index+self.len_feature*j]
                    if value!=0:
                        tag=self.tags[j]
                        fw.write(i.encode('UTF-8')+'\t' + str(value) + '*'+str(tag)+'\n')

    def test_sentence(self, sentence, right_tag):
        match = 0
        total = 0
        #max_tag = self.get_max_score(sentence)
        max_tag = self.get_max_score_average(sentence)
        # for i,j in max_tag,right_tag:
        for index in range(0, len(max_tag)):
            i = max_tag[index]
            j = right_tag[index]
            if i == j:
                match += 1
            total += 1
        return match, total

    def testdata(self,dataset):
        match = 0
        total = 0
        if dataset=='train':
            sentences=self.sentences
            words=self.words
            pos=self.pos
        elif dataset=='dev':
            sentences=self.sentences_dev
            words=self.words_dev
            pos=self.pos_dev
        for i in range(0,len(words)):
            m,t=self.test_sentence(words[i],pos[i])
            match += m
            total += t
        accuracy = match * 1.0 / total
        print 'Right:' + str(match) + 'Total:' + str(total) + 'Accuracy:' + str(accuracy)
        with open('result2.txt', 'a') as fr:
            fr.write(dataset + 'Right:' + str(match) + 'Total:' + str(total) + 'Accuracy:' + str(
                accuracy) + '\n')
        return accuracy

    def test(self, filename):
        match = 0
        total = 0
        with open(filename, 'r') as fh:
            temp_sen = []
            temp_tags = []
            for line in fh:
                if len(line) > 1:
                    words_line = line.strip().split('\t')
                    temp_sen.append(words_line[1].decode('UTF-8'))
                    temp_tags.append(words_line[3])
                else:
                    m, t = self.test_sentence(temp_sen, temp_tags)
                    match += m
                    total += t
                    temp_sen = []
                    temp_tags = []
        accuracy=match * 1.0 / total
        return accuracy
        print 'Right:' + str(match) + 'Total:' + str(total) + 'Accuracy:' + str(accuracy)
        with open('result2.txt', 'a') as fr:
            fr.write(filename + 'Right:' + str(match) + 'Total:' + str(total) + 'Accuracy:' + str(
                accuracy) + '\n')


if __name__ == '__main__':
    glm = GlobalLinearModel()
    glm.readdata()
    glm.create_feature_space()
    glm.perceptron_online_training()
    glm.output()
