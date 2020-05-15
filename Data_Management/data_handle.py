
# -*- encoding: utf-8 -*-
# @author : bobby
# @time : 2020/2/26 14:55

'''
readme
将情感得分用直方图显示出来
用以分析在个得分段中，情感得分的分布情况
'''

import jieba
from wordcloud import WordCloud
from snownlp import SnowNLP
from nltk import *
import pandas as pd
import re
import jieba.posseg as psg
import numpy as np
import pdb
import time
import itertools
import pymysql
from gensim import corpora, models
import matplotlib.pyplot as plt
import filmname# 前端传入的电影名
COUNT = 1000 #读取数据的数量 太多处理时间会很长



jieba.load_userdict("user_dict.txt")
class Data_precess:

    def __init__(self): #初始化
        #初始化数据库连接
        self.db = pymysql.connect(host="localhost", port=3306, user="root", passwd='867425', db="weibofilm",charset="utf8")
        # 使用 cursor() 方法创建一个游标对象 cursor
        self.cursor = self.db.cursor()

    def textParse(self, str_doc):
        # 正则过滤掉特殊符号、标点、英文、数字等。
        r1 = '[a-zA-Z0-9’!"#$%&\'()*+,-./:：;；|<=>?@，—。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'

        # 去除空格
        r2 = '\s+'
        # 去除换行符
        str_doc = re.sub(r1, ' ', str_doc)
        # 多个空格成1个
        str_doc = re.sub(r2, ' ', str_doc)

        # 去掉字符
        str_doc = re.sub('\u3000', '', str_doc)
        return str_doc

    def get_stop_words(self): #加载停用词表
        # pdb.set_trace()
        file = open('NLPIR_stopwords.txt', 'r', encoding='utf-8').read().split('\n')
        return set(file)

    def rm_tokens(self,words, stwlist):#清洗去噪
        words_list = list(words)
        stop_words = stwlist
        for i in range(words_list.__len__())[::-1]:
            if words_list[i] in stop_words:  # 去除停用词
                words_list.pop(i)
            elif words_list[i].isdigit():  # 去除数字
                words_list.pop(i)
            elif len(words_list[i]) == 1:  # 去除单个字符
                words_list.pop(i)
            elif words_list[i] == " ":  # 去除空字符
                words_list.pop(i)
        return words_list

    def read(self): # weibo_essay
        '''
        将comment数据从数据库中读出
        :return:
        '''

        sql ="select id,comment_content,score,content_type from comment where label = '{}'".format(filmname.filmname)
        # sql = 'select content from weibo_essay where id<500 '
        self.cursor.execute(sql)
        temp = self.cursor.fetchall()# 获取结果集列表
        return temp

    def data_to_csv(self,Dict = {}):  #将每句话的评分写入csv
        df = pd.DataFrame(data=Dict,columns=['id','comment','score','content_type'])  # 列标
        df = df.dropna()
        df = df.drop_duplicates(['comment']) #去重
        df.to_csv('comment_score.csv',encoding='utf-8',columns=['id','comment','score'],index=False) #行标

    def sentiment_score(self,str=[]):
        sen = []
        for i in str:
            temp = SnowNLP(i)
            score = temp.sentiments
            sen.append(score)
        return sen

    def bar_draw(self):
        '''
        柱状图
        '''

        sentiments_totle = 'comment_score.csv'
        data = pd.read_csv(sentiments_totle)

        # 列索引
        bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        labels = ['[0,0.1)', '[0.1,0.2)', '[0.2,0.3)', '[0.3,0.4)',
                  '[0.4,0.5)', '[0.5,0.6)', '[0.6,0.7)', '[0.7,0.8)', '[0.8,0.9)', '[0.9,1)']
        data['sentiment_score_layer'] = pd.cut(data.score, bins, labels=labels)
        # pdb.set_trace()
        aggR = data.groupby(by=['sentiment_score_layer'])['score'].agg({'score': np.size})
        pAgg = round(aggR / aggR.sum(), 2, )

        # 绘图
        plt.figure(figsize=(10, 6))  # 设置画布长、宽
        pAgg['score'].plot(kind='bar', width=0.6, fontsize=16)
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 正常显示中文标签
        plt.title('情感得分分布直方图', fontsize=20)
        plt.ylabel('得分百分比')  # y轴标题
        # plt.show()
        plt.savefig('../Text_analysis/static/scrapy/bar.png')

    def box_draw(self):
        '''
        箱线图，查看是否有异常值
        :return:
        '''
        data = pd.read_csv('comment_score.csv')

        plt.rcParams['font.sans-serif'] = ['SimHei']  # 导入图像库
        plt.rcParams['axes.unicode_minus'] = False  # 避免中文显示乱码
        plt.figure()  # 正常显示负号
        x = pd.DataFrame(data['score'])
        x.boxplot()
        # pdb.set_trace()
        plt.show()
        plt.savefig('../Text_analysis/static/scrapy/box.png')

    def cloud_draw(self):
        '''
        词云，查看词频分布
        :return:
        '''
        df = pd.read_csv('comment_score.csv')
        comment = df['comment']

        # map内置高阶函数:一个函数f和list，函数f依次作用在list.
        comment = map(self.textParse, comment)  # 正则处理，去掉一些字符，例如\u3000

        # 2 获取停用词
        stwlist = self.get_stop_words()

        # 3 分词并去除停用词
        word_2dlist = [self.rm_tokens(jieba.cut(part, cut_all=False), stwlist) for part in comment]

        # 4 合并列表
        word_list = sum(word_2dlist, [])

        # frequencies = result.groupby(by=['word'])['word'].count()
        # frequencies = frequencies.sort_values(ascending=False)
        backgroud_Image = plt.imread('pl.jpg')
        wordcloud = WordCloud(font_path="qihei55.ttf",
                              max_words=100,
                              background_color='white',
                              mask=backgroud_Image)
        fdist = FreqDist(word_list)
        r = sorted(fdist.items(), key=lambda item: item[1], reverse=True)
        df = pd.DataFrame(r,columns=['word','count'])
        frequencies = df.groupby(by=['word'])['word'].count()
        frequencies = frequencies.sort_values(ascending=False)
        my_wordcloud = wordcloud.fit_words(frequencies)
        plt.imshow(my_wordcloud)
        plt.axis('off')
        # plt.show()
        # pdb.set_trace()
        plt.savefig('../Text_analysis/static/scrapy/film_cloud.png')

    def pie_draw(self): #饼状图,并得到评论平均值
        df = pd.read_csv('comment_score.csv')
        pos_count = df.loc[df['score'] > 0.6, 'score'].count()  #
        neg_count = df.loc[df['score']<=0.6,'score'].count()
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        labels = ['正向评论数','负向评论数']
        sizes = [pos_count,neg_count]
        explode = (0, 0.1)
        plt.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', shadow=False, startangle=150)
        plt.title("电影正负评论数")
        # plt.show()
        plt.savefig('../Text_analysis/static/scrapy/pie.png')

    def insert(self):
        '''
        将数据存入数据库,
        包括 电影名 评分平均数 爬取数据总量
        :return:
        '''
        df = pd.read_csv('comment_score.csv')
        # pdb.set_trace()
        score_mean = df['score'].mean()
        comment_count = df['comment'].count()
        sql = "select * from film_info where filmname = '{}'".format(filmname.filmname)
        # pdb.set_trace()
        self.cursor.execute(sql)
        if len(self.cursor.fetchall())!=0:  #当前电影名已存在
            return
        else:
            sql = "insert into film_info(filmname, mean, info_totle) values(%s, %s, %s)"  # 注意不要使用关键字

            film =  [str(filmname.film),float(score_mean),int(comment_count)]
            self.cursor.execute(sql,film)
            self.db.commit()

    def change_insert(self,trand):
        '''
                将数据存入Django数据库,
                包括 电影名 评分平均数 爬取数据总量
                :return:
                '''
        db1 = pymysql.connect(host="localhost", port=3306, user="root", passwd='867425', db="text_analysis",
                                  charset="utf8")
        # 使用 cursor() 方法创建一个游标对象 cursor
        cursor1 = db1.cursor()

        #获取电影类型
        sql = "select types from film_type where filmname = '{}'".format(filmname.filmname)
        self.cursor.execute(sql)
        ty = self.cursor.fetchone()[0]

        # pdb.set_trace()
        '''
        电影详情存入Django数据库
        '''
        sql = "select * from getinfo_film where film_name = '{}'".format(filmname.filmname)
        # pdb.set_trace()
        cursor1.execute(sql)
        if len(cursor1.fetchall()) != 0:  # 当前电影名已存在
            sql = "delete from getinfo_film where film_name = '{}'".format(filmname.filmname) #删除原数据
            cursor1.execute(sql)
            db1.commit()

        sql = "insert into getinfo_film(film_name,scrapy_totle,mean,pos,neg,types) values (%s,%s,%s,%s,%s,%s)"
        df = pd.read_csv('comment_score.csv')
        # pdb.set_trace()
        score_mean = round(float(df['score'].mean()),4) #保留四位小数点
        comment_count = int(df['comment'].count())
        info = [filmname.filmname,comment_count,score_mean,trand[0],trand[1],ty]
        cursor1.execute(sql,info)
        db1.commit()

        '''
        评论详情存入数据库
        '''
        sql = "select add_time,comment_user,comment_content,score,label from comment"
        self.cursor.execute(sql)
        temp = self.cursor.fetchall()

        sql = "insert into handleinfo_film_info(add_time,comment_user,comment_content,comment_score,filmname) values (%s,%s,%s,%s,%s)"
        cursor1.executemany(sql,temp)
        db1.commit()

    def LAD_model(self,temp): #建模处理

        #代码1-1 评论去重的代码
        reviews = pd.DataFrame(temp, columns=['id', 'content', 'score', 'content_type'])
        reviews = reviews[['content', 'content_type']].drop_duplicates()
        content = reviews['content']
        # pdb.set_trace()

        # 代码1-2 数据清洗
        # 去除去除英文、数字等
        strinfo = re.compile('[0-9a-zA-Z]|电影|{}|'.format(filmname.filmname))
        content = content.apply(lambda x: strinfo.sub('', x))

        # 代码1-3 分词、词性标注、去除停用词代码
        # 分词
        worker = lambda s: [(x.word, x.flag) for x in psg.cut(s)]  # 自定义简单分词函数
        seg_word = content.apply(worker)

        # 将词语转为数据框形式，一列是词，一列是词语所在的句子ID，最后一列是词语在该句子的位置
        n_word = seg_word.apply(lambda x: len(x))  # 每一评论中词的个数

        n_content = [[x + 1] * y for x, y in zip(list(seg_word.index), list(n_word))]
        index_content = sum(n_content, [])  # 将嵌套的列表展开，作为词所在评论的id

        seg_word = sum(seg_word, [])
        word = [x[0] for x in seg_word]  # 词

        nature = [x[1] for x in seg_word]  # 词性

        content_type = [[x] * y for x, y in zip(list(reviews['content_type']), list(n_word))]
        content_type = sum(content_type, [])  # 评论类型

        result = pd.DataFrame({"index_content": index_content,
                               "word": word,
                               "nature": nature,
                               "content_type": content_type})

        # 删除标点符号
        result = result[result['nature'] != 'x']  # x表示标点符号

        # 删除停用词
        stop_path = open("./data/stoplist.txt", 'r', encoding='UTF-8')
        stop = stop_path.readlines()
        stop = [x.replace('\n', '') for x in stop]
        word = list(set(word) - set(stop))
        result = result[result['word'].isin(word)]

        # 构造各词在对应评论的位置列
        n_word = list(result.groupby(by=['index_content'])['index_content'].count())
        index_word = [list(np.arange(0, y)) for y in n_word]
        index_word = sum(index_word, [])  # 表示词语在改评论的位置

        # 合并评论id，评论中词的id，词，词性，评论类型
        result['index_word'] = index_word

        # 提取含有名词类的评论
        ind = result[['n' in x for x in result['nature']]]['index_content'].unique()
        result = result[[x in ind for x in result['index_content']]]



        #
        word = result
        pos_comment = pd.read_csv("./data/正面评价词语（中文）.txt", header=None, sep="\n",
                                  encoding='utf-8', engine='python')
        neg_comment = pd.read_csv("./data/负面评价词语（中文）.txt", header=None, sep="\n",
                                  encoding='utf-8', engine='python')
        pos_emotion = pd.read_csv("./data/正面情感词语（中文）.txt", header=None, sep="\n",
                                  encoding='utf-8', engine='python')
        neg_emotion = pd.read_csv("./data/负面情感词语（中文）.txt", header=None, sep="\n",
                                  encoding='utf-8', engine='python')

        # 合并情感词与评价词
        positive = set(pos_comment.iloc[:, 0]) | set(pos_emotion.iloc[:, 0])
        negative = set(neg_comment.iloc[:, 0]) | set(neg_emotion.iloc[:, 0])
        intersection = positive & negative  # 正负面情感词表中相同的词语
        positive = list(positive - intersection)
        negative = list(negative - intersection)
        positive = pd.DataFrame({"word": positive,
                                 "weight": [1] * len(positive)})
        negative = pd.DataFrame({"word": negative,
                                 "weight": [-1] * len(negative)})

        posneg = positive.append(negative)

        #  将分词结果与正负面情感词表合并，定位情感词
        data_posneg = posneg.merge(word, left_on='word', right_on='word',
                                   how='right')
        data_posneg = data_posneg.sort_values(by=['index_content', 'index_word'])

        # 根据情感词前时候有否定词或双层否定词对情感值进行修正
        # 载入否定词表
        notdict = pd.read_csv("./data/not.csv")

        # 处理否定修饰词
        data_posneg['amend_weight'] = data_posneg['weight']  # 构造新列，作为经过否定词修正后的情感值
        data_posneg['id'] = np.arange(0, len(data_posneg))
        only_inclination = data_posneg.dropna()  # 只保留有情感值的词语
        only_inclination.index = np.arange(0, len(only_inclination))
        index = only_inclination['id']

        for i in np.arange(0, len(only_inclination)):
            review = data_posneg[data_posneg['index_content'] ==
                                 only_inclination['index_content'][i]]  # 提取第i个情感词所在的评论
            review.index = np.arange(0, len(review))
            affective = only_inclination['index_word'][i]  # 第i个情感值在该文档的位置
            if affective == 1:
                ne = sum([i in notdict['term'] for i in review['word'][affective - 1]])
                if ne == 1:
                    data_posneg['amend_weight'][index[i]] = - \
                        data_posneg['weight'][index[i]]
            elif affective > 1:
                ne = sum([i in notdict['term'] for i in review['word'][[affective - 1,
                                                                        affective - 2]]])
                if ne == 1:
                    data_posneg['amend_weight'][index[i]] = - \
                        data_posneg['weight'][index[i]]

        # 更新只保留情感值的数据
        only_inclination = only_inclination.dropna()

        # 计算每条评论的情感值
        emotional_value = only_inclination.groupby(['index_content'],
                                                   as_index=False)['amend_weight'].sum()

        # 去除情感值为0的评论
        emotional_value = emotional_value[emotional_value['amend_weight'] != 0]

        # 查看情感分析效果
        # 给情感值大于0的赋予评论类型（content_type）为pos,小于0的为neg
        emotional_value['a_type'] = ''
        emotional_value['a_type'][emotional_value['amend_weight'] > 0] = 'pos'
        emotional_value['a_type'][emotional_value['amend_weight'] < 0] = 'neg'

        # 查看情感分析结果
        result = emotional_value.merge(word,
                                       left_on='index_content',
                                       right_on='index_content',
                                       how='left')

        result = result[['index_content', 'content_type', 'a_type']].drop_duplicates()
        confusion_matrix = pd.crosstab(result['content_type'], result['a_type'],
                                       margins=True)  # 制作交叉表
        (confusion_matrix.iat[0, 0] + confusion_matrix.iat[1, 1]) / confusion_matrix.iat[2, 2]

        # 提取正负面评论信息
        ind_pos = list(emotional_value[emotional_value['a_type'] == 'pos']['index_content'])
        ind_neg = list(emotional_value[emotional_value['a_type'] == 'neg']['index_content'])
        posdata = word[[i in ind_pos for i in word['index_content']]]
        negdata = word[[i in ind_neg for i in word['index_content']]]

        #主题词为长度大于一的名词
        posdata = posdata[posdata['nature'] == 'n']
        posdata = posdata[posdata['word'].map(len)>1]

        negdata = negdata[negdata['nature'] == 'n']
        negdata = negdata[negdata['word'].map(len) > 1]

        #  建立情感词典
        # 建立词典
        pos_dict = corpora.Dictionary([[i] for i in posdata['word']])  # 正面
        neg_dict = corpora.Dictionary([[i] for i in negdata['word']])  # 负面

        # 建立语料库
        pos_corpus = [pos_dict.doc2bow(j) for j in [[i] for i in posdata['word']]]  # 正面
        neg_corpus = [neg_dict.doc2bow(j) for j in [[i] for i in negdata['word']]]  # 负面

        # 主题数寻优
        # 构造主题数寻优函数
        def cos(vector1, vector2):  # 余弦相似度函数
            dot_product = 0.0;
            normA = 0.0;
            normB = 0.0;
            for a, b in zip(vector1, vector2):
                dot_product += a * b
                normA += a ** 2
                normB += b ** 2
            if normA == 0.0 or normB == 0.0:
                return (None)
            else:
                return (dot_product / ((normA * normB) ** 0.5))

            # 主题数寻优
        def lda_k(x_corpus, x_dict):
            # 初始化平均余弦相似度
            mean_similarity = []
            mean_similarity.append(1)

            # 循环生成主题并计算主题间相似度
            for i in np.arange(2, 11):
                lda = models.LdaModel(x_corpus, num_topics=i, id2word=x_dict)  # LDA模型训练
                for j in np.arange(i):
                    term = lda.show_topics(num_words=50)

                # 提取各主题词
                top_word = []
                for k in np.arange(i):
                    top_word.append([''.join(re.findall('"(.*)"', i)) \
                                     for i in term[k][1].split('+')])  # 列出所有词

                # 构造词频向量
                word = sum(top_word, [])  # 列出所有的词
                unique_word = set(word)  # 去除重复的词

                # 构造主题词列表，行表示主题号，列表示各主题词
                mat = []
                for j in np.arange(i):
                    top_w = top_word[j]
                    mat.append(tuple([top_w.count(k) for k in unique_word]))

                p = list(itertools.permutations(list(np.arange(i)), 2))
                l = len(p)
                top_similarity = [0]
                for w in np.arange(l):
                    vector1 = mat[p[w][0]]
                    vector2 = mat[p[w][1]]
                    top_similarity.append(cos(vector1, vector2))

                # 计算平均余弦相似度
                mean_similarity.append(sum(top_similarity) / l)
            return (mean_similarity)

        # 计算主题平均余弦相似度
        pos_k = lda_k(pos_corpus, pos_dict)
        neg_k = lda_k(neg_corpus, neg_dict)


        from matplotlib.font_manager import FontProperties
        font = FontProperties(size=14)
        # 解决中文显示问题
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        fig = plt.figure(figsize=(10, 8))
        ax1 = fig.add_subplot(211)
        ax1.plot(pos_k)
        ax1.set_xlabel('正面评论LDA主题数寻优', fontproperties=font)

        ax2 = fig.add_subplot(212)
        ax2.plot(neg_k)
        ax2.set_xlabel('负面评论LDA主题数寻优', fontproperties=font)


        # LDA主题分析

        pos_lda = models.LdaModel(pos_corpus, num_topics=3, id2word=pos_dict)
        neg_lda = models.LdaModel(neg_corpus, num_topics=3, id2word=neg_dict)
        # 最终建模数据

        pos = ''
        for i in pos_lda.print_topics(num_words=10):
            pos = pos+i[1]+'+'
        neg = ''
        for i in neg_lda.print_topics(num_words=10):
            neg = neg+i[1]+'+'
        trend = []
        trend.append(pos)
        trend.append(neg)  #将情感主题加入list 等待存储入数据库
        return trend


    def trytest(self):
        s = pd.read_csv('data/posdata.csv')
        pdb.set_trace()

if __name__ == '__main__':
    process = Data_precess()
    temp = process.read()
    process.data_to_csv(temp) # 将数据存入csv
    process.bar_draw()#
    process.box_draw()
    process.cloud_draw()
    process.pie_draw()
    process.insert()
    trand = process.LAD_model(temp)
    process.change_insert(trand)
    # process.trytest()
    # pdb.set_trace()



