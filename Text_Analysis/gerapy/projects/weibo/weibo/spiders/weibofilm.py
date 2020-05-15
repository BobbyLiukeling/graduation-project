# -*- coding: utf-8 -*-
'''
readme
里是没调微博的详情页面
通过连接爬取评论里的内容
'''
import scrapy

from selenium import webdriver
from selenium.webdriver import ActionChains
import time
import pymysql
import re
from urllib import parse
import pdb
import traceback, sys
from scrapy.selector import Selector
from ..items import CommentItem,WeiboItem
from .urls import temp
import os
from .filmname import filmname
from snownlp import SnowNLP
from decimal import Decimal


class WeibofilmSpider(scrapy.Spider):
    name = 'weibofilm'
    os.system('python "../labels.py"') # 将上次爬取的链接进行处理
    # allowed_domains = ['www.weibo.com']

    # file = open('urls.txt','r')

    path = []
    for i in temp:
        i = i.split('/')[-1]

        path.append('https://m.weibo.cn/detail/' + i)
    start_urls = path[0:]#有评论
    def __init__(self,**kwargs):
        self.db = pymysql.connect(host="localhost", port=3306, user="root", passwd='867425', db="weibofilm",charset="utf8")


        # 使用 cursor() 方法创建一个游标对象 cursor
        self.cursor = self.db.cursor()
        self.web = webdriver.Chrome("E:/software/python3.6/chromedriver.exe")


        #点击qq登陆微博账号
        #随便打开一个微博网页
        # pdb.set_trace()
        # self.web.get('https://m.weibo.cn/detail/4375850843355381')
        # time.sleep(3)
        # js = "var q=document.documentElement.scrollTop=100000"  # 向下加载出登陆界面
        # self.web.execute_script(js)
        # time.sleep(3)
        self.web.get('https://m.weibo.cn/quicklogin?r=https%3A%2F%2Fm.weibo.cn%2Fdetail%2F4375850843355381')
        time.sleep(2)

        # pdb.set_trace()
        #点击QQ登陆
        # ac = self.web.find_element_by_xpath(".//div[@class = 'share-icon']/a")
        ac = self.web.find_element_by_xpath(".//li[@class = 'l-uitem']/a")
        self.web.execute_script("arguments[0].click();", ac)  # 用js执行
        time.sleep(3)

        #点击QQ头像,先要获取iframe，再使用正常的方式
        self.web.switch_to.frame(self.web.find_element_by_xpath(".//iframe[@id = 'ptlogin_iframe']"))#进入iframe,获取未加载源码
        ac = self.web.find_element_by_xpath(".//span[@id = 'img_out_1194380923']")#id根据QQ号决定
        self.web.execute_script("arguments[0].click();", ac)  # 用js执行
        time.sleep(3)

    @staticmethod
    def close(spider, reason):
        closed = getattr(spider, 'closed', None)
        if callable(closed):
            print('* ' * 50)
            return closed(reason)

    def parse(self, response):
        '''
        微博内容
        '''
        item = WeiboItem()
        self.web.get(response.url)
        time.sleep(2)
        page = self.web.page_source
        label = filmname
        try:
            user = Selector(text=page).xpath(".//h3[@class = 'm-text-cut']/text()").extract()[0].strip().strip('\n')#去掉空格和换行符
        #     content = Selector(text=page).xpath(".//div[@class = 'weibo-text']").extract()[0]
            fav_nums = Selector(text=page).xpath(".//div[@class = 'lite-page-tab']/div[3]/i[2]/text()").extract()[0]
            if '万' in fav_nums:
                fav_nums = float(fav_nums[:-1])*10000
            comment_nums = Selector(text=page).xpath(".//div[@class = 'lite-page-tab']/div[2]/i[2]/text()").extract()[0]
            if '万' in comment_nums:
                comment_nums = float(comment_nums[:-1])*10000
        #     forward_nums = Selector(text=page).xpath(".//div[@class = 'lite-page-tab']/div[1]/i[2]/text()").extract()[0]
        #     if '万' in forward_nums:
        #         forward_nums = float(forward_nums[:-1])*10000
        #     add_time = Selector(text=page).xpath(".//span[@class = 'time']/text()").extract()[0]
        #
        #     item['label'] = label
        #     item['user'] = user
        #     item['content'] = content
        #     item['fav_nums'] = int(fav_nums)
        #     item['comment_nums'] = int(comment_nums)
        #     item['forward_nums'] = int(forward_nums)
        #     item['add_time'] = add_time
        #     # pdb.set_trace()
        #     yield item
        except Exception as e:
            print(e)
            pass


        '''
        微博评论
        '''
        # f = open('extent_comment.txt', 'a')
        item_comment = CommentItem()
        try:
            '''评论内容外键'''
            sql = 'select id from weibo_essay where user = "{0}" and fav_nums = {1}'.format(user, int(fav_nums))
            self.cursor.execute(sql)
            essay_id = self.cursor.fetchone()[0]
        except:
            essay_id = 0
        label_comment = filmname
        try:
            if int(comment_nums == 0): # 没用评论
                pass
            elif int(comment_nums)>20 :#评论数大于20需要向下加载
                #每2000像素滑动条长度加载20条评论
                count = len(Selector(text=self.web.page_source).xpath(".//div[@class = 'comment-content']/div").extract())
                # pdb.set_trace()
                #将所有评论都加载出来
                for i in range(2, int(int(comment_nums)/20)+1):  # 也可以设置一个较大的数，一下到底
                    js = "var q=document.documentElement.scrollTop={}".format(i * 3500)  # javascript语句,第一个不用加载了
                    self.web.execute_script(js)
                    time.sleep(2)
                    count_temp = len(Selector(text=self.web.page_source).xpath(".//div[@class = 'comment-content']/div").extract())
                    if count < count_temp:#向下加载出了新的评论
                         count = count_temp
                    else:#加载到了底部
                        break#跳出当前循环
                page_comment = Selector(text=self.web.page_source).xpath(".//div[@class = 'comment-content']/div").extract()
                time.sleep(2)
                for i in page_comment[:-1]:  # 注意去掉最后一个
                    try:#评论下多条回复
                        Selector(text=i).xpath(".//div[@class = 'cmt-sub-txt']/p[3]/span/text()").extract()[0]
                    except:#评论下无回复
                        user_comment = Selector(text=i).xpath(".//div[@class = 'm-text-box']/h4/text()").extract()[0]
                        time.sleep(0.2)
                        try:
                            content_comment = Selector(text=i).xpath(".//div[@class = 'm-text-box']/h3/text()").extract()[0]
                        except:
                            content_comment = '回复为图片'  # 评论回复为图片
                        fav_nums_comment = Selector(text=i).xpath(".//div[contains(@class,'lite-bot') and contains(@class,'m-text-cut')]/aside[2]/em/text()").extract()
                        if len(fav_nums_comment) == 0:  # 没有点赞
                            fav_nums_comment = 0
                        else:
                            fav_nums_comment = fav_nums_comment[0]
                            if '万' in fav_nums_comment:
                                fav_nums_comment = float(fav_nums_comment[:-1]) * 10000
                        add_time_comment = Selector(text=i).xpath(".//div[contains(@class,'lite-bot') and contains(@class,'m-text-cut')]/div/text()").extract()[0].strip().strip('\n')
                        item_comment['label'] = label_comment
                        item_comment['essay_id'] = essay_id
                        item_comment['user'] = user_comment
                        item_comment['fav_nums'] = fav_nums_comment
                        item_comment['add_time'] = add_time_comment
                        item_comment['content'] = content_comment
                        item_comment['score'] = self.get_score(content_comment)
                        if float(item_comment['score']) > 0.5:
                            item_comment['type'] = 'pos'
                        else:
                            item_comment['type'] = 'neg'
                        yield item_comment
                time.sleep(2)


            else :#直接爬取未超过20, 评论下面没有回复的直接爬取
                page_comment = Selector(text=self.web.page_source).xpath(".//div[@class = 'comment-content']/div").extract()

                # pdb.set_trace()
                time.sleep(2)
                for i in page_comment[:-1]: #注意去掉最后一个
                    # user =  label = essay_id = fav_nums = content = add_time =
                    user_comment = Selector(text=i).xpath(".//div[@class = 'm-text-box']/h4/text()").extract()[0]
                    time.sleep(0.3)
                    try:
                        content_comment = Selector(text=i).xpath(".//div[@class = 'm-text-box']/h3/text()").extract()[0]
                    except:
                        content_comment = '回复为图片'#评论回复为图片
                    fav_nums_comment = Selector(text=i).xpath(".//div[contains(@class,'lite-bot') and contains(@class,'m-text-cut')]/aside[2]/em/text()").extract()
                    if len(fav_nums_comment)==0:#没有点赞
                        fav_nums_comment = 0
                    else:
                        fav_nums_comment = fav_nums_comment[0]
                    add_time_comment = Selector(text=i).xpath(".//div[contains(@class,'lite-bot') and contains(@class,'m-text-cut')]/div/text()").extract()[0].strip().strip('\n')
                    # try:#评论下的回复
                    #                     #     content_comment = Selector(text=i).xpath( ".//div[@class = 'cmt-sub-txt']/p/span/text()").extract()[1]#评论下的回复，一律不点开，只抓第一条
                    #                     #     item_comment['label'] = label_comment
                    #                     #     item_comment['essay_id'] = essay_id
                    #                     #     item_comment['user'] = user_comment
                    #                     #     item_comment['fav_nums'] = fav_nums_comment
                    #                     #     item_comment['add_time'] = add_time_comment
                    #                     #     item_comment['content'] = content_comment
                    #                     #     yield item_comment
                    #                     # except:
                    #                     #     pass
                    item_comment['label'] = label_comment
                    item_comment['essay_id'] = essay_id
                    item_comment['user'] = user_comment
                    item_comment['fav_nums'] = fav_nums_comment
                    item_comment['add_time'] = add_time_comment
                    item_comment['content'] = content_comment
                    item_comment['score'] = self.get_score(content_comment)
                    if float(item_comment['score'])>0.5:
                        item_comment['type'] = 'pos'
                    else:
                        item_comment['type'] = 'neg'
                    yield item_comment

        except Exception as e:
            print(e)
            pass


    def get_score(self,str):
        score = SnowNLP(str)
        s = score.sentiments
        return float(round(s,4))








