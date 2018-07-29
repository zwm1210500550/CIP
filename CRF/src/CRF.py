import numpy as np
import datetime
import random
from config import config
from collections import defaultdict
from scipy.misc import logsumexp

def data_handle(filename):
    if filename=='None':
        return False
    sentences = list()
    sentence = list()
    sentence_num = 0
    word_num = 0
    with open(filename,"r") as dataTxt:
        for line in dataTxt:
            if len(line) == 1:
                sentences.append(sentence)
                sentence = []
                sentence_num += 1
            else:
                word = line.split()[1]
                tag = line.split()[3]
                sentence.append((word,tag))
                word_num += 1
    print("文件%s:共%d个句子，%d个词" % (filename,sentence_num,word_num))
    return sentences    #sentences格式：[[(戴相龙,NR),(,),(,)....],[],[]....]


class CRF(object):
    def __init__(self,train_data_file,dev_data_file,test_data_file):
        self.train_data = data_handle(train_data_file)
        self.dev_data = data_handle(dev_data_file)
        self.test_data = data_handle(test_data_file)
        tags = set()
        for sentence in self.train_data:
            for word,tag in sentence:
                tags.add(tag)
        self.tags = list(tags)      #tag的列表
        self.N = len(self.tags)     #tag的数目
        self.index_to_tag = {index:tag for index,tag in enumerate(self.tags)}
        self.tag_to_index = {tag:index for index,tag in enumerate(self.tags)}

    def create_bigram_feature(self,pre_tag,cur_tag):
        return ["01:" + cur_tag + "*" + pre_tag] 

    def create_unigram_feature(self,sentence,index,cur_tag):
        words = [_word for _word,_tag in sentence]
        words.insert(0,"^")     #插入句首开始符
        words.append("$")       #插入句尾结束符
        unigram_feature = []
        unigram_feature.append("02:" + cur_tag + "*" + words[index+1])
        unigram_feature.append("03:" + cur_tag + "*" + words[index])
        unigram_feature.append("04:" + cur_tag + "*" + words[index+2])
        unigram_feature.append("05:" + cur_tag + "*" + words[index+1] + "*" + words[index][-1])
        unigram_feature.append("06:" + cur_tag + "*" + words[index+1] + "*" + words[index+2][0])
        unigram_feature.append("07:" + cur_tag + "*" + words[index+1][0])
        unigram_feature.append("08:" + cur_tag + "*" + words[index+1][-1])
        for j in range(1,len(words[index+1])-1):
            unigram_feature.append("09:" + cur_tag + "*" + words[index+1][j])
            unigram_feature.append("10:" + cur_tag + "*" + words[index+1][0] + "*" + words[index+1][j])
            unigram_feature.append("11:" + cur_tag + "*" + words[index+1][-1] + "*" + words[index+1][j])
        if len(words[index+1]) == 1:
            unigram_feature.append("12:" + cur_tag + "*" + words[index+1] + "*" + words[index][-1] + "*" + words[index+2][0])
        for j in range(len(words[index+1])-1):
            if words[index+1][j] == words[index+1][j+1]:
                unigram_feature.append("13:" + cur_tag + "*" + words[index+1][j] + "*" + "consecutive")
        for j in range(1,min(len(words[index+1]),4)+1):
            unigram_feature.append("14:" + cur_tag + "*" + words[index+1][0:j])
            unigram_feature.append("15:" + cur_tag + "*" + words[index+1][-j:])
        return unigram_feature
        
    def create_feature_template(self,sentence,index,pre_tag,cur_tag):    #创建某个tag相对于sentence句子中第index个词的特征
        feature_template = self.create_bigram_feature(pre_tag,cur_tag) + self.create_unigram_feature(sentence,index,cur_tag)
        return feature_template

    def create_feature_space(self):
        feature_space_list = list()
        for sentence in self.train_data:
            pre_tag = "start"
            for i in range(len(sentence)):
                cur_tag = sentence[i][1]
                feature_space_list += self.create_feature_template(sentence,i,pre_tag,cur_tag)
                pre_tag = cur_tag 
        self.feature_space_list = list(set(feature_space_list))
        self.feature_space = {feature:index for index,feature in enumerate(self.feature_space_list)}    #特征空间是一个字典，格式：{“NR*戴相龙:0”,"",....}
        self.E = len(self.feature_space)    #特征空间的数目
        self.bigram_features = [[self.create_bigram_feature(pre_tag,cur_tag) for pre_tag in self.tags] for cur_tag in self.tags] 
        self.bigram_scores = np.zeros((self.N, self.N))

    def score(self,features):
        scores = [self.w[self.feature_space[feature]] for feature in features if feature in self.feature_space]
        return sum(scores) 

    def predict(self,sentence):
        length = len(sentence)
        delta = np.zeros((length,self.N)) 
        path = np.zeros((length,self.N),dtype=int)
        for i in range(self.N):
            features_first = self.create_feature_template(sentence,0,"start", self.index_to_tag[i])
            delta[0][i] = self.score(features_first)
        path[0] = -1
        for i in range(1,length):
            unigram_scores = np.array([self.score(self.create_unigram_feature(sentence,i,tag)) for tag in self.tags])
            scores = self.bigram_scores + unigram_scores[:,None] + delta[i-1]
            path[i] = np.argmax(scores,axis=1)
            delta[i] = np.max(scores,axis=1)
        predict_tag_list = list()
        tag_index = np.argmax(delta[length-1])
        predict_tag_list.append(self.index_to_tag[tag_index])
        for i in range(length-1):
            tag_index = path[length-1-i][tag_index]
            predict_tag_list.insert(0,self.index_to_tag[tag_index])
        return predict_tag_list
        
    def evaluate(self,sentences):
        count_right = 0
        count_all = 0
        for sentence in sentences:
            right_tag_list = [word_tag[1] for word_tag in sentence]
            predict_tag_list = self.predict(sentence)
            length = len(right_tag_list)
            count_all += length 
            for i in range(length):
                if right_tag_list[i] == predict_tag_list[i]:
                    count_right += 1
        precision = count_right/count_all
        print("正确词数：%d\t总词数：%d\t正确率%f" % (count_right,count_all,precision))
        return precision 

    def forward(self, sentence):
        length = len(sentence)
        logalpha = np.zeros((length, self.N))  
        logalpha[0] = [self.score(self.create_feature_template(sentence, 0, "start", cur_tag)) for cur_tag in self.tags]
        for i in range(1, length):
            unigram_scores = np.array([self.score(self.create_unigram_feature(sentence, i, cur_tag)) for cur_tag in self.tags])
            scores = self.bigram_scores + unigram_scores[:,None] + logalpha[i-1]
            logalpha[i] = logsumexp(scores, axis=1)
        return logalpha 

    def backward(self, sentence):
        length = len(sentence)
        logbeta = np.zeros((length, self.N)) 
        for i in reversed(range(length-1)):
            unigram_scores = np.array([self.score(self.create_unigram_feature(sentence, i+1, cur_tag)) for cur_tag in self.tags])
            scores = self.bigram_scores.T + unigram_scores + logbeta[i+1]
            logbeta[i] = logsumexp(scores, axis=1)
        return logbeta

    def update_gradient(self, sentence, tag_list):
        length = len(sentence)
        pre_tag = "start"
        for i in range(length):
            cur_tag = tag_list[i]
            features = self.create_feature_template(sentence, i, pre_tag, cur_tag)
            for feature in features:
                self.g[self.feature_space[feature]] += 1
            pre_tag = cur_tag 
        logalpha = self.forward(sentence)
        logbeta = self.backward(sentence)
        dinominator = logsumexp(logalpha[-1])
        for index, tag in self.index_to_tag.items():
            features = self.create_feature_template(sentence, 0, "start", tag)
            feature_index = [self.feature_space[feature] for feature in features if feature in self.feature_space]
            prob = np.exp(self.score(features) + logbeta[0][index] - dinominator)
            for id in feature_index:
                self.g[id] -= prob
        for i in range(1, length):
            for index, tag in self.index_to_tag.items():
                unigram_features = self.create_unigram_feature(sentence, i, tag)
                unigram_index = [self.feature_space[feature] for feature in unigram_features if feature in self.feature_space]
                scores = self.bigram_scores[index] + self.score(unigram_features) + logalpha[i-1] + logbeta[i][index]
                probs = np.exp(scores - dinominator)
                for bigram_features, prob in zip(self.bigram_features[index], probs):
                    bigram_index = [self.feature_space[bigram_feature] for bigram_feature in bigram_features if bigram_feature in self.feature_space]
                    for feature_index in unigram_index + bigram_index:
                        self.g[feature_index] -= prob 

        

            
    def SGD_training(self,iterator=100,stop_iterator=10,batch_size=1, regularization=False, step_opt=False, C=0.01, eta=1.0):
        self.w = np.zeros(self.E)
        self.g = defaultdict(float)
        b = 0
        learn_rate = eta
        decay_rate = 0.96
        decay_steps = len(self.train_data) / batch_size 
        global_step = 1
        max_dev_data_precision = 0
        max_dev_data_precision_index = 0
        if regularization:
            print("使用正则化：C=%f" % C)
        else:
            print("不使用正则化")
        if step_opt:
            print("使用模拟退火：eta=%f" % eta)
        else:
            print("不使用模拟退火：eta=%f" %eta)
        for iter in range(iterator):
            print("第%d次迭代：" % (iter+1))
            startime = datetime.datetime.now()
            print("正在打乱训练数据...")
            random.shuffle(self.train_data)
            print("数据已打乱")
            if regularization:
                print("使用正则化：C=%f" % C)
            if step_opt:
                print("使用模拟退火：learn_rate=%f" % learn_rate)
            for sentence in self.train_data:
                b += 1
                right_tag_list = [word_tag[1] for word_tag in sentence]
                self.update_gradient(sentence, right_tag_list)
                if b == batch_size:
                    if regularization:
                        self.w *= (1 - C * eta)
                    for id, value in self.g.items():
                        self.w[id] += value *  learn_rate 
                    if step_opt:
                        learn_rate = eta * decay_rate ** (global_step / decay_steps)
                    global_step += 1
                    b = 0
                    self.g = defaultdict(float)
                    self.bigram_scores = np.array([[self.score(bigram_feature) for bigram_feature in bigram_features] for bigram_features in self.bigram_features])
            print("训练集：",end="")
            train_data_precision = self.evaluate(self.train_data)
            print("开发集：",end="")
            dev_data_precision = self.evaluate(self.dev_data)
            if dev_data_precision > max_dev_data_precision:
                now_train_data_precision = train_data_precision 
                max_dev_data_precision = dev_data_precision
                max_dev_data_precision_index = iter + 1
                if self.test_data:
                    print("测试集：",end="")
                    test_data_precision = self.evaluate(self.test_data)
            stoptime = datetime.datetime.now()
            time = stoptime - startime 
            print("本轮用时：%s" % str(time))
            if ((iter+1)-max_dev_data_precision_index) > stop_iterator:     #stop_iterator轮性能没有提升
                break
        print("\n共迭代%d轮" % (iter+1))
        print("开发集第%d轮准确率最高:" % max_dev_data_precision_index)
        print("此时训练集准确率:%f" % now_train_data_precision)
        print("此时开发集准确率:%f" % max_dev_data_precision)
        if self.test_data:
            print("此时测试集准确率为:%f" % test_data_precision)


if __name__ == "__main__":
    train_data_file = config['train_data_file']
    dev_data_file = config['dev_data_file']
    test_data_file = config['test_data_file']
    iterator = config['iterator']
    stop_iterator = config['stop_iterator']
    batch_size = config['batch_size']
    regularization = config['regularization']
    step_opt = config['step_opt']
    C = config['C']
    eta = config['eta']


    startime = datetime.datetime.now()
    lm = CRF(train_data_file,dev_data_file,test_data_file)
    lm.create_feature_space()
    lm.SGD_training(iterator, stop_iterator, batch_size, regularization, step_opt, C, eta)
    stoptime = datetime.datetime.now()
    time = stoptime - startime
    print("耗时：" +  str(time))

