import numpy as np
import datetime
import random
import math
from scipy.misc import logsumexp 
from collections import defaultdict
from config import config

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
decay_steps = config['decay_steps']

def data_handle(filename):
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


class log_linear_model(object):
    def __init__(self):
        self.train_data = data_handle(train_data_file)
        self.dev_data = data_handle(dev_data_file)
        self.test_data = data_handle(test_data_file)
        tags = set()
        for sentence in self.train_data:
            for word,tag in sentence:
                tags.add(tag)
        self.tags = list(tags)      
        self.tags_dic = {tag:index for index,tag in enumerate(self.tags)}
        self.N = len(self.tags)     #tag的数目

    
    def create_feature_template(self,sentence,index):    #创建sentence句子中第index个词的部分特征模板
        words = [_word for _word,_tag in sentence]
        words.insert(0,"^")     #插入句首开始符
        words.append("$")       #插入句尾结束符
        feature_template = set()
        feature_template.add("02:" + words[index+1])
        feature_template.add("03:" + words[index])
        feature_template.add("04:" + words[index+2])
        feature_template.add("05:" + words[index+1] + "*" + words[index][-1])
        feature_template.add("06:" + words[index+1] + "*" + words[index+2][0])
        feature_template.add("07:" + words[index+1][0])
        feature_template.add("08:" + words[index+1][-1])
        for j in range(1,len(words[index+1])-1):
            feature_template.add("09:" + words[index+1][j])
            feature_template.add("10:" + words[index+1][0] + "*" + words[index+1][j])
            feature_template.add("11:" + words[index+1][-1] + "*" + words[index+1][j])
        if len(words[index+1]) == 1:
            feature_template.add("12:" + words[index+1] + "*" + words[index][-1] + "*" + words[index+2][0])
        for j in range(len(words[index+1])-1):
            if words[index+1][j] == words[index+1][j+1]:
                feature_template.add("13:" + words[index+1][j] + "*" + "consecutive")
        for j in range(1,min(len(words[index+1]),4)+1):
            feature_template.add("14:" + words[index+1][0:j])
            feature_template.add("15:" + words[index+1][-j:])
        return feature_template

    def create_feature_space(self):
        feature_space = set()
        for sentence in self.train_data:
            for i in range(len(sentence)):
                feature_space |= self.create_feature_template(sentence,i)
        self.feature_space_list = list(feature_space)
        self.feature_space = {feature:index for index,feature in enumerate(self.feature_space_list)}    #特征空间是一个字典，格式：{“NR*戴相龙:0”,"",....}
        self.E = len(self.feature_space)    #特征空间的数目

    def dot(self, features):
        scores = np.array([self.w[self.feature_space[feature]] for feature in features if feature in self.feature_space])
        return np.sum(scores, axis=0)

    def SGD_training(self):
        self.w = np.zeros((self.E,self.N))  
        g = defaultdict(float)
#        g = np.zeros((self.N,self.E))  
        b = 0
        global_step = 0
        decay_rate = 0.96
        learn_rate = eta 
        max_dev_data_precision = 0
        max_dev_data_precision_index = 0
        for iter in range(iterator):
            print("\n第%d次迭代：" % (iter+1))
            startime = datetime.datetime.now()
            print("正在打乱训练数据...")
            random.shuffle(self.train_data)
            print("数据已打乱")
            if step_opt:
                print("使用步长优化")
                print('learn_rate:%f' % learn_rate)
            else:
                print("不使用步长优化")
            if regularization:
                print("使用正则化")
                print('正则化系数C:%f' % C)
            else:
                print("不使用正则化")

            for sentence in self.train_data:
                for i in range(len(sentence)):
                    right_tag = sentence[i][1]
                    features = self.create_feature_template(sentence,i)
                    scores = self.dot(features)
                    scores_all = logsumexp(scores)
                    prob = np.exp(scores-scores_all)
                    for feature in features:
                        g[self.feature_space[feature]] -= prob
                        g[(self.feature_space[feature],self.tags_dic[right_tag])] += 1


                    """
                    scores = np.zeros(self.N)
                    for feature in features:
                        g[(self.tags_dic[right_tag],self.feature_space[feature])] += 1
                    for tag, tag_index in self.tags_dic.items():
                        for feature in features:
                            scores[tag_index] += self.w[tag_index][self.feature_space[feature]]
                    scores_all = logsumexp(scores)
                    prob = np.exp(scores - scores_all)
                    for tag , tag_index in self.tags_dic.items():
                        for feature in features:
                            g[(tag_index,self.feature_space[feature])] -= prob[tag_index]
                    """
                    b += 1
                    
                    if b==batch_size:
                        if regularization:
                            self.w *= (1 - C*learn_rate)
                        
                        for id ,value in g.items():
                            self.w[id] += learn_rate*value

#                        self.w +=learn_rate * g
                            
                        if step_opt:
                            learn_rate = eta*decay_rate ** (global_step / decay_steps)
                        b = 0
                        global_step += 1
                        g = defaultdict(float)
#                        g = np.zeros((self.N,self.E))
                        
            print("训练集：",end="")
            train_data_precision = self.evaluate(self.train_data)
            print("开发集：",end="")
            dev_data_precision = self.evaluate(self.dev_data)
            if dev_data_precision > max_dev_data_precision:
                now_train_data_precision = train_data_precision 
                max_dev_data_precision = dev_data_precision
                max_dev_data_precision_index = iter + 1
                print("测试集：",end="")
                test_data_precision = self.evaluate(self.test_data)
            stoptime = datetime.datetime.now()
            time = stoptime - startime 
            print("本轮用时：%s" % str(time))
            if ((iter+1)-max_dev_data_precision_index) > stop_iterator:     #stop_iterator轮性能没有提升
                break
        print("\n共迭代%d轮" % (iter+1))
        print("开发集第%d轮准确率最高:" % max_dev_data_precision_index)
        print("此时训练集准确率为:%f" % now_train_data_precision)
        print("此时开发集准确率为:%f" % max_dev_data_precision)
        print("此时测试集准确率为:%f" % test_data_precision)

    def evaluate(self,sentences):
        count_right = 0
        count_all = 0
#        score_matrix = np.zeros(self.N,dtype=int)
        for sentence in sentences:
            for i in range(len(sentence)):
                count_all += 1
                right_tag = sentence[i][1]
                features = self.create_feature_template(sentence,i)
                scores = self.dot(features)
                max_tag = self.tags[np.argmax(scores)]
                if right_tag == max_tag:
                    count_right += 1
        precision = count_right/count_all
        print("正确词数：%d\t总词数：%d\t正确率%f" % (count_right,count_all,precision))
        return precision 



if __name__ == "__main__":
    startime = datetime.datetime.now()
    lm = log_linear_model()
    lm.create_feature_space()
    lm.SGD_training()
    stoptime = datetime.datetime.now()
    time = stoptime - startime
    print("耗时：" +  str(time))

