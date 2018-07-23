#!\usr\bin\python
# coding=UTF-8

class linear_model:
    sentences = []  # 句子的列表
    pos = []  # words对应的词性
    words = []  # 句子分割成词的词组列表的列表
    dic_feature = {}  # 特征向量的字典
    dic_tags = {}  # 词性的集合
    def _init_(self):
        self.sentences=[] # 句子的列表
        self.pos=[] # words对应的词性
        self.words=[] # 句子分割成词的词组列表的列表
        self.dic_feature={} # 特征向量的字典
        self.dic_tags={} # 词性的集合

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

    def create_feature_templates(self,words,index,tag):
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
        f.append('02:'+str(tag)+'*'+wos[index])
        f.append('03:'+str(tag)+'*'+wi_minus)
        f.append('04:'+str(tag)+'*'+wi_plus)
        f.append('05:'+str(tag)+'*'+wos[index]+'*'+ci_minus_minus1)
        f.append('06:'+str(tag)+'*'+wos[index]+ci_plus0)
        f.append('07:'+str(tag)+'*'+ci0)
        f.append('08:'+str(tag)+'*'+ci_minus1)
        for i in range(1,len(wos[index])-1):
            f.append('09:'+str(tag)+'*'+wos[index][i])
            f.append('10:'+str(tag)+'*'+wos[index][0]+'*'+wos[index][i])
            f.append('11:'+str(tag)+'*'+wos[index][-1]+'*'+wos[index][i])
            if wos[index][i]==wos[index][i+1]:
                f.append('13:'+str(tag)+'*'+wos[index][i]+'*'+'consecutive')
        if len(wos[index])==1:
            f.append('12:'+str(tag)+'*'+ci_minus_minus1+'*'+ci_plus0)
        for i in range(1,len(wos[index])+1):
            if i>4:
                break
            f.append('14:'+str(tag)+'*'+wos[index][0:i+1])
            f.append('15:'+str(tag)+'*'+wos[index][-i-1:-1])
        return f

    def create_feature_space(self):
        for i in range(0,len(self.sentences)):
            for j in range(0,len(self.words[i])):
                f=self.create_feature_templates(self.words[i],j,self.pos[i][j])
                for feat in f:
                    self.dic_feature[feat]=0
                if self.pos[i][j] in self.dic_tags:
                    self.dic_tags[self.pos[i][j]]+=1
                else:
                    self.dic_tags[self.pos[i][j]] = 0
        print('特征空间维度：%d'%len(self.dic_feature))
        print ('词性维度：%d'%len(self.dic_tags))

    def get_score(self,f):
        score=0
        for i in f :
            if i in self.dic_feature:
                score+=self.dic_feature[i]
        return score

    def get_max_tag(self,words,index):
        max_score=-10
        tag='NULL'
        for t in self.dic_tags:
            f=self.create_feature_templates(words,index,t)
            temp_score=self.get_score(f)
            if temp_score>max_score:
                max_score=temp_score
                tag=t
        return tag

    def online_training(self):
        for iteration in range(0,20):
            print '当前迭代：'+str(iteration+1)
            for index_sen in range(0,len(self.sentences)):
                for index_word in range(0,len(self.words[index_sen])):
                    #f=self.create_feature_templates()
                    max_tag=self.get_max_tag(self.words[index_sen],index_word)
                    right_tag=self.pos[index_sen][index_word]
                    if max_tag!=right_tag:
                         f_right_tag=self.create_feature_templates(self.words[index_sen],index_word,right_tag)
                         f_max_tag=self.create_feature_templates(self.words[index_sen],index_word,max_tag)
                         for i in f_right_tag:
                             if i in self.dic_feature:
                                 self.dic_feature[i]+=1
                         for i in f_max_tag:
                             if i in self.dic_feature:
                                 self.dic_feature[i]-=1
            self.test('train.conll')

    def output(self):
        with open('model.txt','w+') as fm:
            for i in self.dic_feature:
                fm.write(i.encode('utf-8')+'\t'+str(self.dic_feature[i])+'\n')
        print 'Output Successfully'

    def test_sentence(self,words,tags):
        right=0
        sumup=0
        for i in range(0,len(words)):
            max_tag=self.get_max_tag(words,i)
            sumup+=1
            if max_tag==tags[i]:
                right+=1
        return right,sumup

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
        print('正确：'+str(right)+'总数：'+str(total)+'正确率:'+str(pricision))
        with open('result.txt','a+') as fr:
            fr.write(filename+ '正确：'+str(right)+'总数：'+str(total)+'正确率:'+str(pricision)+'\n')

if __name__=='__main__':
    lm = linear_model()
    lm.readfile()
    lm.create_feature_space()
    lm.online_training()
    lm.output()
    lm.test('dev.conll')
