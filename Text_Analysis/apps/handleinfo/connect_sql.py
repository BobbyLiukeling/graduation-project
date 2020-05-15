# -*- encoding: utf-8 -*-
# @author : bobby
# @time : 2020/2/24 21:22
'''
将爬虫数据库中文件传入Django数据库中
'''
import pymysql
import xlsxwriter
import pdb
class Data_read:

    def __init__(self):
        self.db1 = pymysql.connect(host="localhost", port=3306, user="root", passwd='867425', db="weibofilm",charset="utf8")
        # 使用 cursor() 方法创建一个游标对象 cursor
        self.cursor1 = self.db1.cursor()
        self.db2 = pymysql.connect(host="localhost", port=3306, user="root", passwd='867425', db="text_analysis",
                                   charset="utf8")
        # 使用 cursor() 方法创建一个游标对象 cursor
        self.cursor2 = self.db2.cursor()
        self.db1.autocommit(1)
        self.db1.autocommit(2)

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

