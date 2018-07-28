#!\usr\bin\python
# coding=UTF-8
import math
import numpy as np

class log_linear_model:
    def __init__(self):
        self.sentences = []  # 句子的列表
        self.pos = []  # words对应的词性
        self.words = []  # 句子分割成词的词组列表的列表
        self.feature = {}  # 特征向量的字典
        self.dic_tags = {}  # 词性的集合
        self.tags = []  # 词性的列表
        self.len_tags = 0
        self.len_feature = 0
        self.weight = []  # 特征向量值
        self.g=[] # 临时特征向量值

    def readfile(self):
        with open("train.conll",'r') as ft:
            temp_words = []
            temp_sen = ''
            temp_poss = []
            for line in ft:
                if len(line)>1:
                    temp_word=line.strip().split('\t')[1].decode('utf-8')
                    temp_pos=line.strip().split('\t')[3]
                    temp_sen+=temp_word
                    temp_words.append(temp_word)
                    temp_poss.append(temp_pos)
                else:
                    self.words.append(temp_words)
                    self.sentences.append(temp_sen)
                    self.pos.append(temp_poss)
                    temp_words = []
                    temp_sen = ''
                    temp_poss = []

    def create_feature_templates(self,words,index):
        f=[]
        #words_of_sentence
        wos=words
        if index==0:
            wi_minus='$$'
            ci_minus_minus1='$'
        else:
            wi_minus = wos[index - 1]
            ci_minus_minus1 = wi_minus[-1]

        if index==len(wos)-1:
            wi_plus='##'
            ci_plus0='#'
        else:
            wi_plus = wos[index + 1]
            ci_plus0 = wi_plus[0]
        ci0=wos[index][0]
        ci_minus1=wos[index][-1]
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
                f.append('13:'+ wos[index][i] + '*' + 'consecutive')
        if len(wos[index]) == 1:
            f.append('12:' + ci_minus_minus1 + '*' + ci_plus0)
        for i in range(1, len(wos[index]) + 1):
            if i > 4:
                break
            f.append('14:'+ wos[index][0:i+1])
            f.append('15:'+ wos[index][-i - 1:-1])
        return f

    def create_feature_space(self):
        for i in range(0, len(self.sentences)):
            for j in range(0, len(self.words[i])):
                f = self.create_feature_templates(self.words[i], j)
                for feat in f:
                    if feat not in self.feature:
                        self.feature[feat] = len(self.feature)
                if self.pos[i][j] not in self.tags:
                    self.tags.append(self.pos[i][j])
        self.len_feature = len(self.feature)
        self.len_tags = len(self.tags)
        self.dic_tags = {value: index for index, value in enumerate(self.tags)}
        self.weight = np.zeros(self.len_tags*self.len_feature)
        self.g = np.zeros(self.len_tags * self.len_feature)
        print('特征空间维度：%d' % len(self.feature))
        print ('词性维度：%d' % len(self.dic_tags))

    def get_score(self,f,offset):
        score=0
        for i in f :
            if i in self.feature:
                score+=self.weight[self.feature[i]+offset]
        return score

    def get_max_tag(self,words,index):
        max_score=-10
        tag='NULL'
        for i in range(0,self.len_tags):
            f=self.create_feature_templates(words,index)
            temp_score=self.get_score(f,i*self.len_feature)
            if temp_score>max_score:
                max_score=temp_score
                tag=self.tags[i]
        return tag

    def get_prob(self,words,index):
        down=0
        matr=np.zeros(self.len_tags)
        for i in range(0, self.len_tags):
            f=self.create_feature_templates(words,index)
            temp_score=math.exp(self.get_score(f,i*self.len_feature))
            down+=temp_score
            matr[i]=temp_score
        matr=matr/down
        return matr

    def SGD_training(self):
        B = 50
        b = 0
        k = 0
        C = 0.0001

        global_step = 1
        decay_rate = 0.96
        decay_steps = 100000
        eta = 0.5
        learn_rate = eta

        for iteration in range(0,20):
            print '当前迭代次数'+str(iteration+1)
            for index_sen in range(0,len(self.sentences)):
                for index_word in range(0,len(self.words[index_sen])):
                    tag=self.pos[index_sen][index_word]
                    f_tag=self.create_feature_templates(self.words[index_sen],index_word)
                    for i in f_tag:
                        if i in self.feature:
                            self.g[self.feature[i]+self.dic_tags[tag]*self.len_feature]+=1
                    prob=self.get_prob(self.words[index_sen],index_word)
                    for i_tag in range(0,self.len_tags):
                        for i in f_tag:
                            if i in self.feature:
                                self.g[self.feature[i] + i_tag * self.len_feature] -= prob[i_tag]
                    b=b+1
                    if B == b:
                        self.weight*= (1 - C * learn_rate)
                        self.weight+=eta*self.g
                        k += 1
                        b = 0
                        self.g = np.zeros(self.len_tags * self.len_feature)
                        learn_rate = eta * decay_rate ** (global_step / decay_steps)
                        global_step += 1

            self.test('train.conll')
            self.test('dev.conll')
        print '模型更新次数' + str(k)

    def output(self):
        with open('model2.txt', 'w+') as fm:
            for i_tag in range(0,self.len_tags):
                for i in self.feature:
                    index=self.feature[i]+i_tag*self.len_feature
                    myweight=self.weight[index]
                    if myweight!=0:
                        fm.write(i.encode('utf-8') + self.tags[i_tag] + '\t'+str(myweight) + '\n')
        print 'Output Successfully'

    def test_sentence(self,words,tags):
        right=0
        for i in range(0,len(words)):
            max_tag=self.get_max_tag(words,i)
            if max_tag==tags[i]:
                right+=1
        return right,len(words)

    def test(self,filename):
        right=0
        total=0
        with open(filename,'r') as ft:
            temp_words=[]
            temp_pos=[]
            for line in ft:
                if len(line)>1:
                    str_line=line.strip().split('\t')
                    temp_words.append(str_line[1].decode('utf-8'))
                    temp_pos.append(str_line[3])
                else:
                    sen_right,sen_len=self.test_sentence(temp_words,temp_pos)
                    right+=sen_right
                    total+=sen_len
                    temp_words=[]
                    temp_pos=[]
        pricision=1.0*right/total
        print '正确：'+str(right)+'总数：'+str(total)+'正确率:'+str(pricision)
        with open('result_norm2_2.txt','a+') as fr:
            fr.write(filename +'\t'+'正确：'+str(right)+'总数：'+str(total)+'正确率:'+str(pricision)+'\n')


if __name__ == '__main__':
    llm = log_linear_model()
    llm.readfile()
    llm.create_feature_space()
    llm.SGD_training()
    llm.test('dev.conll')
    llm.output()
