# -*- coding: utf-8 -*-

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
from .filmname import filmname
url = []

class FilmSpider(scrapy.Spider):
    name = 'film'
    start_urls = ['https://m.weibo.cn/search?containerid=100103type%3D1%26q%3D{}'.format(filmname)]


    def __init__(self,**kwargs):
        self.db = pymysql.connect(host="localhost", port=3306, user="root", passwd='867425', db="weibofilm",
                             charset="utf8")

        # 使用 cursor() 方法创建一个游标对象 cursor
        self.cursor = self.db.cursor()
        self.web = webdriver.Chrome("E:/software/python3.6/chromedriver.exe")

    @staticmethod
    def close(spider, reason):
        closed = getattr(spider, 'closed', None)
        if callable(closed):
            print('* ' * 50)
            return closed(reason)

    def parse(self, response):
        self.web.get(response.url)#加载此链接网页
        time.sleep(2)
        self.web.find_element_by_xpath(".//ul[contains(@class,'nav-item') and contains(@class,'center')]/li/span[text() = '话题']").click()#点击话题栏
        time.sleep(2)

        '''
        加载话题
        '''
        js = "var q=document.documentElement.scrollTop=100000" #100000加载10个话题
        self.web.execute_script(js)
        time.sleep(1)
        js = "var q=document.documentElement.scrollTop=100000"  # 100000加载10个话题
        self.web.execute_script(js)
        time.sleep(1)
        js = "var q=document.documentElement.scrollTop=100000"  # 100000加载10个话题
        self.web.execute_script(js)
        time.sleep(1)
        js = "var q=document.documentElement.scrollTop=100000"  # 100000加载10个话题
        self.web.execute_script(js)
        time.sleep(3)



        url_lev_1 = response.url#第一层搜索--话题链接
        count_scrollTop = len(Selector(text=self.web.page_source).xpath(".//div[@class = 'card card11']")) #加载出的 第一层总量

        f = open('try.txt', 'a')
        try:
            for t in range(0, count_scrollTop):#会向下加载更多 加载出来的数量
                count_topic = len(Selector(text=self.web.page_source).xpath(".//div[@class = 'card card11']")[t].xpath(".//div[@class = 'card-list']/div"))
                # pdb.set_trace()
                # pdb.set_trace()
                # count_topic = len(Selector(text=self.web.page_source).xpath(".//div[@class = 'card-list']/div"))
                try:
                    for i in range(1,count_topic):#进入大话题
                        # pdb.set_trace()
                        ac_lev_2 = self.web.find_element_by_xpath(".//div[@class = 'card card11'][%s]"%(t+1)).find_element_by_xpath(".//div[@class = 'card-list']/div[%s]/div/div/div" % i)
                        # ac_lev_2 = self.web.find_element_by_xpath((".//div[@class = 'card-list']/div[%s]/div/div/div"%i))
                        self.web.execute_script("arguments[0].click();", ac_lev_2)  # 用js执行
                        time.sleep(3)
                        js = "var q=document.documentElement.scrollTop=40000"
                        self.web.execute_script(js)
                        time.sleep(3)
                        count_easy = len(Selector(text=self.web.page_source).xpath(".//div[@class = 'm-container-max']/div/div/div"))#话题下总共的文章个数
                        url_lev_2 = self.web.current_url
                        for j in range(4,count_easy):
                            try:#中间会有很多是不能点开的
                                ac_lev_3 = self.web.find_element_by_xpath(".//div[@class = 'm-container-max']/div/div/div[%s]" % j).find_element_by_xpath(".//footer/div[2]/h4")
                                self.web.execute_script("arguments[0].click();", ac_lev_3)  # 用js执行
                                time.sleep(1)
                                url.append(self.web.current_url)
                                f.write(self.web.current_url)
                                f.write('$')
                                self.web.get(url_lev_2)  # 回到上层页面
                                time.sleep(3)
                            except Exception as e:
                                print(e)
                                print(20*'j ')
                                # pdb.set_trace()
                                pass
                        self.web.get(url_lev_1)  # 回到上层页面
                        time.sleep(3)
                        self.web.find_element_by_xpath( ".//ul[contains(@class,'nav-item') and contains(@class,'center')]/li/span[text() = '话题']").click()  # 点击话题栏
                        time.sleep(3)
                        #分开加载
                        js = "var q=document.documentElement.scrollTop=100000"  # 100000加载10个话题
                        self.web.execute_script(js)
                        time.sleep(1)
                        js = "var q=document.documentElement.scrollTop=100000"  # 100000加载10个话题
                        self.web.execute_script(js)
                        time.sleep(1)
                        js = "var q=document.documentElement.scrollTop=100000"  # 100000加载10个话题
                        self.web.execute_script(js)
                        time.sleep(1)
                        js = "var q=document.documentElement.scrollTop=100000"  # 100000加载10个话题
                        self.web.execute_script(js)
                        time.sleep(3)
                    time.sleep(5)
                except Exception as e:
                    print(20* 'i')
                    print(e)
                    pass
        except Exception as e:
            print(e)
            print( 20*'t ')
            pass
        f.close()
        for w in url:
            print(w)
        # try:
        #     for i in range(1,count_topic):#进入大话题
        #         # if i%7 == 0 :
        #         #     # pdb.set_trace()
        #         #     js = "var q=document.documentElement.scrollTop={}".format(i*120)
        #         #     self.web.execute_script(js)
        #         #     # self.web.execute_script("var q=document.documentElement.scrollTop={}".format(i*200))
        #         #     time.sleep(1)
        #         self.web.find_element_by_xpath(".//div[@class = 'card-list']/div[%s]/div/div/div"%i).click()  # 从url_topic进到下一层（第二层）页面
        #         time.sleep(3)
        #         # js = "var q=document.documentElement.scrollTop=40000"
        #         # self.web.execute_script(js)
        #         # time.sleep(6)
        #         count_easy = len(Selector(text=self.web.page_source).xpath(".//div[@class = 'm-container-max']/div/div/div"))#话题下总共的文章个数
        #         url_lev_2 = self.web.current_url
        #         for j in range(4,count_easy):
        #             try:#中间会有很多是不能点开的
        #                 ac = self.web.find_element_by_xpath(".//div[@class = 'm-container-max']/div/div/div[%s]" % j).find_element_by_xpath(".//footer/div[2]/h4")
        #                 self.web.execute_script("arguments[0].click();", ac)  # 用js执行
        #                 time.sleep(1)
        #                 url.append(self.web.current_url)
        #                 self.web.get(url_lev_2)  # 回到上层页面
        #                 time.sleep(3)
        #             except Exception as e:
        #                 print(e)
        #                 # pdb.set_trace()
        #                 pass
        #
        #             # href = Selector(text=self.web.page_source).xpath(".//div[@class = 'm-container-max']/div/div/div[$temp]/div/div/article/div/div/a[text() = '全文']/@href",temp=j).extract()
        #             # pdb.set_trace()
        #             # self.web.find_element_by_xpath(".//div[@class = 'm-container-max']/div/div/div[%s]/div/div/footer/div[2]/h4"%j).click()
        #             # ac = self.web.find_element_by_xpath(".//div[@class = 'm-container-max']/div/div/div[%s]/div/div/footer/div[2]/h4"%j)
        #             # ac = self.web.find_element_by_xpath(".//div[@class = 'm-container-max']/div/div/div[5]/div/div/footer/div[2]/h4")
        #             # ActionChains(self.web).move_to_element(ac).click(ac).perform()
        #             # class = 'card m-panel card9 weibo-member card-vip'
        #             # Selector(text=self.web.page_source).xpath(".//div[@class = 'm-container-max']/div/div/div[5]/div/div/footer/div[2]/h4").extract()
        #             # class = 'card card11'
        #             # Selector(text=self.web.page_source).xpath(".//div[@class = 'm-container-max']/div/div/div[5]/div/div/footer/div[2]/h4").extract()
        #             # 'm-ctrl-box m-box-center-a' 'm-ctrl-box m-box-center-a'
        #             # self.web.find_element_by_xpath( ".//footer[contains(@class,'m-ctrl-box') and contains(@class,'m-box-center-a')]/div[2]/h4").click()  # 点击话题栏
        #             # Selector(text=self.web.page_source).xpath(".//footer[contains(@class,'m-ctrl-box') and contains(@class,'m-box-center-a')]/div[2]/h4")[1].extract()
        #             # if len(href)== 0:# 没有展开全文
        #             #    pass #之后再写
        #             # else:
        #             #     href = href[0].split('/')[2]
        #             #     url.append(href)
        #
        #         #回到topic画面
        #         # pdb.set_trace()
        #         self.web.get(url_lev_1)  # 回到上层页面
        #         time.sleep(3)
        #         self.web.find_element_by_xpath( ".//ul[contains(@class,'nav-item') and contains(@class,'center')]/li/span[text() = '话题']").click()  # 点击话题栏
        #         time.sleep(3)








