# -*- encoding: utf-8 -*-
# @author : bobby
# @time : 2020/1/19 20:00
import csv
import xlsxwriter
import pymysql
import pdb
import json
import time
import re
import jieba 
from nltk import *
import pandas as pd
import matplotlib
import sys
from sklearn.feature_extraction.text import TfidfTransformer,CountVectorizer
from snownlp import SnowNLP as sn
from snownlp import sentiment
# sentiment.train('neg.txt','pos.txt') #情感语料训练库

jieba.load_userdict("user_dict.txt") #词频划分时使用
#机器学习，以数据为基础，进行归纳和总结
#模型：数据解释现象的系统


class Data_read:

    def __init__(self):
        self.db = pymysql.connect(host="localhost", port=3306, user="root", passwd='867425', db="weibofilm",charset="utf8")
        # 使用 cursor() 方法创建一个游标对象 cursor
        self.cursor = self.db.cursor()

    @staticmethod
    def close(self):
        self.cursor.close()

    def read(self): # weibo_essay
        '''
        将comment数据从数据库中读出
        :return:
        '''
        sql ="select comment_content from comment "
        # sql = 'select content from weibo_essay where id<500 '
        self.cursor.execute(sql)
        temp = self.cursor.fetchall()# 获取结果集列表
        return temp

    def filter_Htmltags(self,htmlstr):
        # 过滤DOCTYPE
        htmlstr = ' '.join(htmlstr.split())  # 去掉多余的空格
        re_doctype = re.compile(r'<!DOCTYPE .*?> ', re.S)
        s = re_doctype.sub('', htmlstr)

        # 过滤CDATA
        re_cdata = re.compile('//<!CDATA\[[ >]∗ //\] > ', re.I)
        s = re_cdata.sub('', s)

        # Script
        re_script = re.compile('<\s*script[^>]*>[^<]*<\s*/\s*script\s*>', re.I)
        s = re_script.sub('', s)  # 去掉SCRIPT

        # style
        re_style = re.compile('<\s*style[^>]*>[^<]*<\s*/\s*style\s*>', re.I)
        s = re_style.sub('', s)  # 去掉style

        # 处理换行
        re_br = re.compile('<br\s*?/?>')
        s = re_br.sub('', s)  # 将br转换为换行

        # HTML标签
        re_h = re.compile('</?\w+[^>]*>')
        s = re_h.sub('', s)  # 去掉HTML 标签

        # HTML注释
        re_comment = re.compile('<!--[^>]*-->')
        s = re_comment.sub('', s)

        # 多余的空行
        blank_line = re.compile('\n+')
        s = blank_line.sub('', s)

        blank_line_l = re.compile('\n')
        s = blank_line_l.sub('', s)

        blank_kon = re.compile('\t')
        s = blank_kon.sub('', s)

        blank_one = re.compile('\r\n')
        s = blank_one.sub('', s)

        blank_two = re.compile('\r')
        s = blank_two.sub('', s)

        blank_three = re.compile(' ')
        s = blank_three.sub('', s)

        # 剔除超链接
        http_link = re.compile(r'(http://.+.html)')
        s = http_link.sub('', s)
        return s

    # 正则对字符串清洗
    def textParse(self,str_doc):
        # 正则过滤掉特殊符号、标点、英文、数字等。
        r1 = '[a-zA-Z0-9’!"#$%&\'()*+,-./:：;；|<=>?@，—。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'
        # 去除空格
        r2 = '\s+'
        # 去除换行符
        str_doc = re.sub(r1, ' ', str_doc)
        # 多个空格成1个
        str_doc = re.sub(r2, ' ', str_doc)
        # 去除换行符
        # str_doc = str_doc.replace('\n',' ')

        # 去掉字符
        str_doc = re.sub('\u3000', '', str_doc)
        return str_doc

    # def textParse(self,str_doc):
    #     # 正则过滤掉特殊符号、标点、英文、数字等。
    #     r1 = '%'
    #     str_doc = re.sub(r1, ' ', str_doc)
    #     return str_doc



    # ********************1 结巴中文分词***********************************

    # 利用jieba对文本进行分词，返回切词后的list
    def seg_doc(self,str_doc):
        # 1 正则处理原文本
        sent_list = str_doc.split('\n')
        # map内置高阶函数:一个函数f和list，函数f依次作用在list.
        sent_list = map(self.textParse, sent_list)  # 正则处理，去掉一些字符，例如\u3000

        # 2 获取停用词
        stwlist = self.get_stop_words()

        # 3 分词并去除停用词
        word_2dlist = [self.rm_tokens(jieba.cut(part, cut_all=False), stwlist) for part in sent_list]

        # 4 合并列表
        word_list = sum(word_2dlist, [])
        return word_list


    # 获取创建停用词列表
    def get_stop_words(self):
        # pdb.set_trace()
        file = open('NLPIR_stopwords.txt', 'r', encoding='utf-8').read().split('\n')
        return set(file)

    # 去掉一些停用词和数字
    def rm_tokens(self,words, stwlist):
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





    # 利用nltk进行词频特征统计
    def nltk_wf_feature(self,word_list):
        # ********统计词频方法1**************
        fdist = FreqDist(word_list)
        # print(fdist.keys(), fdist.values())
        # print('=' * 3, '指定词语词频统计', '=' * 3)
        # w = '训练'
        # print(w, '出现频率：', fdist.freq(w))  # 给定样本的频率
        # print(w, '出现次数：', fdist[w])  # 出现次数
        #
        # print('=' * 3, '频率分布表', '=' * 3)
        # fdist.tabulate(30)  # 频率分布表

        # print('='*3,'可视化词频','='*3)
        # fdist.plot(30) # 频率分布图
        # fdist.plot(30,cumulative=True) # 频率累计图

        # print('='*3,'根据词语长度查找词语','='*3)
        # wlist =[w for w in fdist if len(w)>2]
        # print(wlist)

        # ********统计词频方法2**************
        # from collections import Counter
        # Words = Counter(word_list)
        # print(Words.keys(),Words.values())
        # wlist =[w for w in Words if len(w)>2]
        # print(wlist)
        return fdist


    def insert_essay_count(self):
        sql = "insert into comment_count(keyss, nums) values(%s, %s)" #注意不要使用关键字
        # sql = "insert into essay_count(keyss, nums) values(%s, %s)" #注意不要使用关键字
        r = sorted(self.nltk_wf_feature(self.Date_Change()).items(), key=lambda item: item[1],reverse=True) # 将字典值以values进行排序
        # pdb.set_trace()
        try:
            self.cursor.executemany(sql,r)
        except Exception as e:
            print(e)
        finally:
            self.db.commit()
    # def dict_into_list(self,Dict):

    def insert_sentiment_score(self): # 存储文本情感评分
        sql = "insert into weibo_sentiments(keyss, valuess) values(%s, %s)"  # 注意不要使用关键字
        # sql = "insert into essay_count(keyss, nums) values(%s, %s)" #注意不要使用关键字
        r = []
        for i in self.Date_Change().items():
            r.append(i)
        # r = self.Date_Change()
        # pdb.set_trace()
        try:
            self.cursor.executemany(sql, r)
        except Exception as e:
            pdb.set_trace()
            print(e)
        finally:
            self.db.commit()

    def Date_Change(self):
        text = []
        change = {}
        for i in self.read():
            temp = self.filter_Htmltags(i[0])  #去掉HTML元素及标签
            # temp = self.textParse(temp)
            if temp == '':
                continue
            s = sn(temp)
            t = s.sentiments
            change[temp] = t
        # json.dumps(m, ensure_ascii=False).decode('utf8').encode('gb2312')
        # with open('weibo_sentiment.json', 'w',encoding='utf-8') as f: # 将得出的情感得分存储为json文件
        #     w = json.dumps(change, ensure_ascii=False)
        #     f.write(w)
        return change  #情感得分字典

    # def Date_Change(self):
    #     text = []
    #     for i in self.read():
    #         text.append(self.filter_Htmltags(i[0]))  #去掉HTML元素及标签
    #     world = ''.join(text)
    #     world = self.textParse(world) #去除停用词
    #     pdb.set_trace()
    #     word_list = self.seg_doc(world)
    #     dict_date = self.nltk_wf_feature(word_list)#统计词频
    #
    #     return dict_date #词频统计字典

# 功能：将一字典写入到csv文件中
# 输入：文件名称，数据字典
    def Dict_to_csv(self,my_dict={}):
        workbook = xlsxwriter.Workbook('my_dict.xlsx')
        worksheet = workbook.add_worksheet()

        # 设定格式，等号左边格式名称自定义，字典中格式为指定选项
        # bold：加粗，num_format:数字格式
        i = 0
        for item in my_dict.items():
            # 行列表示法的单元格下标以0作为起始值，如‘3,0’等价于‘A3’
            worksheet.write(i, 0, item[0])  # 使用列行表示法写入数字‘123’
            worksheet.write(i, 1, item[1])  # 使用列行表示法写入数字‘456’
            i = i+1
        workbook.close()

    def split_into_excel(self):
        '''
        将每条评论分词结果存入excel中
        :return:
        '''
        workbook = xlsxwriter.Workbook('jieba_word.xlsx')
        worksheet = workbook.add_worksheet()

        # 设定格式，等号左边格式名称自定义，字典中格式为指定选项
        # bold：加粗，num_format:数字格式
        x = self.read()
        j = 0
        # pdb.set_trace()
        for i in x:
            temp = self.filter_Htmltags(i[0]) #去掉HTML标签
            temp = self.textParse(temp) #过滤掉符号
            # pdb.set_trace()
            temp = self.seg_doc(temp)#去掉停用词，结巴分词 ['第一张', '图片']
            temp = ' '.join(temp)
            if len(temp) == 0:
                continue
            worksheet.write(j,0,0)
            worksheet.write(j,1,temp)
            j = j+1
        workbook.close()



if __name__ == '__main__':
    date = Data_read()
    start = time.time()
    date.split_into_excel()
    # date.Date_Change()
    # date.insert_sentiment_score()
    date.insert_essay_count()
    date.Dict_to_csv(date.Date_Change())
    # date.createDictCSV('weibo_sentiment.csv',date.Date_Change())
    end = time.time()
    totle_time = end - start
    print("time_totle: " ,totle_time)
    # date.insert_essay_count()




