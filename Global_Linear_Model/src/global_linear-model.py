import numpy as np
import datetime
import random
from config import config

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


class global_linear_model(object):
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
        bigram_feature = set()
        bigram_feature.add("01:" + cur_tag + "*" + pre_tag)
        return bigram_feature 

    def create_unigram_feature(self,sentence,index,cur_tag):
        words = [_word for _word,_tag in sentence]
        words.insert(0,"^")     #插入句首开始符
        words.append("$")       #插入句尾结束符
        unigram_feature = set()
        unigram_feature.add("02:" + cur_tag + "*" + words[index+1])
        unigram_feature.add("03:" + cur_tag + "*" + words[index])
        unigram_feature.add("04:" + cur_tag + "*" + words[index+2])
        unigram_feature.add("05:" + cur_tag + "*" + words[index+1] + "*" + words[index][-1])
        unigram_feature.add("06:" + cur_tag + "*" + words[index+1] + "*" + words[index+2][0])
        unigram_feature.add("07:" + cur_tag + "*" + words[index+1][0])
        unigram_feature.add("08:" + cur_tag + "*" + words[index+1][-1])
        for j in range(1,len(words[index+1])-1):
            unigram_feature.add("09:" + cur_tag + "*" + words[index+1][j])
            unigram_feature.add("10:" + cur_tag + "*" + words[index+1][0] + "*" + words[index+1][j])
            unigram_feature.add("11:" + cur_tag + "*" + words[index+1][-1] + "*" + words[index+1][j])
        if len(words[index+1]) == 1:
            unigram_feature.add("12:" + cur_tag + "*" + words[index+1] + "*" + words[index][-1] + "*" + words[index+2][0])
        for j in range(len(words[index+1])-1):
            if words[index+1][j] == words[index+1][j+1]:
                unigram_feature.add("13:" + cur_tag + "*" + words[index+1][j] + "*" + "consecutive")
        for j in range(1,min(len(words[index+1]),4)+1):
            unigram_feature.add("14:" + cur_tag + "*" + words[index+1][0:j])
            unigram_feature.add("15:" + cur_tag + "*" + words[index+1][-j:])
        return unigram_feature
        
    def create_feature_template(self,sentence,index,pre_tag,cur_tag):    #创建某个tag相对于sentence句子中第index个词的特征
        feature_template = set()
        feature_template = self.create_bigram_feature(pre_tag,cur_tag) | self.create_unigram_feature(sentence,index,cur_tag)
        return feature_template

    def create_feature_space(self):
        feature_space = set()
        for sentence in self.train_data:
            pre_tag = "start"
            for i in range(len(sentence)):
                cur_tag = sentence[i][1]
                feature_space |= self.create_feature_template(sentence,i,pre_tag,cur_tag)
                pre_tag = cur_tag 
        self.feature_space_list = list(feature_space)
        self.feature_space = {feature:index for index,feature in enumerate(self.feature_space_list)}    #特征空间是一个字典，格式：{“NR*戴相龙:0”,"",....}
        self.E = len(self.feature_space)    #特征空间的数目
        self.bigram_features = [[self.create_bigram_feature(pre_tag,cur_tag) for pre_tag in self.tags] for cur_tag in self.tags] 

    def score(self,features,averaged):
        if averaged:
            scores = [self.v[self.feature_space[feature]] for feature in features if feature in self.feature_space] 
        else:
            scores = [self.w[self.feature_space[feature]] for feature in features if feature in self.feature_space]
        return sum(scores) 


    def predict(self,sentence,averaged):
        length = len(sentence)
        delta = np.zeros((length,self.N)) 
        path = np.zeros((length,self.N),dtype=int)
        for i in range(self.N):
            features_first = self.create_feature_template(sentence,0,"start",self.index_to_tag[i])
            delta[0][i] = self.score(features_first,averaged)
        path[0] = -1
        bigram_scores = np.array([[self.score(_bigram_feature,averaged) for _bigram_feature in bigram_feature] for bigram_feature in self.bigram_features])
        for i in range(1,length):
            unigram_scores = np.array([self.score(self.create_unigram_feature(sentence,i,tag),averaged) for tag in self.tags])
            scores = bigram_scores + unigram_scores[:,None] + delta[i-1]
            path[i] = np.argmax(scores,axis=1)
            delta[i] = np.max(scores,axis=1)
        predict_tag_list = list()
        tag_index = np.argmax(delta[length-1])
        predict_tag_list.append(self.index_to_tag[tag_index])
        for i in range(length-1):
            tag_index = path[length-1-i][tag_index]
            predict_tag_list.insert(0,self.index_to_tag[tag_index])
        return predict_tag_list

            


    def perceptron_online_training(self,averaged,iterator,stop_iterator):
        self.w = np.zeros(self.E,dtype=int)         
        self.v = np.zeros(self.E,dtype=int)
        self.update_times = np.zeros(self.E,dtype=int)
        update_time = 0
        if averaged:
            print("使用累加特征权重：")
        else:
            print("不使用累加特征权重：")
        max_dev_data_precision = 0
        max_dev_data_precision_index = 0
        for iter in range(iterator):
            print("第%d次迭代：" % (iter+1))
            startime = datetime.datetime.now()
            print("正在打乱训练数据...")
            random.shuffle(self.train_data)
            print("数据已打乱")
            for sentence in self.train_data:
                right_tag_list = [word_tag[1] for word_tag in sentence]
                predict_tag_list = self.predict(sentence,False)
                if right_tag_list != predict_tag_list:
                    update_time += 1
                    right_pre_tag = "start"
                    predict_pre_tag = "start"
                    for i in range(len(sentence)):
                        right_cur_tag = right_tag_list[i]
                        predict_cur_tag = predict_tag_list[i]
                        features = self.create_feature_template(sentence,i,right_pre_tag,right_cur_tag)
                        for feature in features:
                            feature_index = self.feature_space[feature]
                            last_w_value = self.w[feature_index]
                            self.w[feature_index] += 1
                            self.update_v(feature_index,last_w_value,update_time)
                        features = self.create_feature_template(sentence,i,predict_pre_tag,predict_cur_tag)
                        for feature in features:
                            if feature in self.feature_space:
                                feature_index = self.feature_space[feature]
                                last_w_value = self.w[feature_index]
                                self.w[self.feature_space[feature]] -= 1
                                self.update_v(feature_index,last_w_value,update_time)
                        right_pre_tag = right_cur_tag
                        predict_pre_tag = predict_cur_tag
            
            #结束本轮迭代，更新全部的v
            for i in range(self.E):
                last_time = self.update_times[i]
                last_w_value = self.w[i]
                self.update_times[i] = update_time
                self.v[i] += (update_time-last_time)*last_w_value



            print("训练集：",end="")
            train_data_precision = self.evaluate(self.train_data,averaged)
            print("开发集：",end="")
            dev_data_precision = self.evaluate(self.dev_data,averaged)
            if dev_data_precision > max_dev_data_precision:
                now_train_data_precision = train_data_precision 
                max_dev_data_precision = dev_data_precision
                max_dev_data_precision_index = iter + 1
                if self.test_data:
                    print("测试集：",end="")
                    test_data_precision = self.evaluate(self.test_data,averaged)
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


    def evaluate(self,sentences,averaged):
        count_right = 0
        count_all = 0
        for sentence in sentences:
            right_tag_list = [word_tag[1] for word_tag in sentence]
            predict_tag_list = self.predict(sentence,averaged)
            length = len(right_tag_list)
            count_all += length 
            for i in range(length):
                if right_tag_list[i] == predict_tag_list[i]:
                    count_right += 1
        precision = count_right/count_all
        print("正确词数：%d\t总词数：%d\t正确率%f" % (count_right,count_all,precision))
        return precision 

    def update_v(self,feature_index,last_w_value,update_time):
        last_time = self.update_times[feature_index]
        self.update_times[feature_index] = update_time
        self.v[feature_index] += (update_time-last_time-1) * last_w_value + self.w[feature_index] 

if __name__ == "__main__":
    train_data_file = config['train_data_file']
    dev_data_file = config['dev_data_file']
    test_data_file = config['test_data_file']
    averaged = config['averaged']
    iterator = config['iterator']
    stop_iterator = config['stop_iterator']


    startime = datetime.datetime.now()
    lm = global_linear_model(train_data_file,dev_data_file,test_data_file)
    lm.create_feature_space()
    lm.perceptron_online_training(averaged,iterator,stop_iterator)
    stoptime = datetime.datetime.now()
    time = stoptime - startime
    print("耗时：" +  str(time))

