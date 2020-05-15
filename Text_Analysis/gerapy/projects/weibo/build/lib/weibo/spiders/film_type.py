# -*- coding: utf-8 -*-
# @author : bobby
# @time : 2020/3/3 18:35

'''
爬取近十年的上线电影名称，以及电影类型，并存入数据库
'''
import scrapy
import pymysql

import pdb
from ..items import Types
# pdb.set_trace()
# from .type_url import urls



class FilmTypeSpider(scrapy.Spider):
    name = 'film_type'

    # temp = []
    # for i in urls:
    #     temp.append(i.strip())
    # start_urls = temp

    # pdb.set_trace()
    strs = 'https://www.1905.com/mdb/film/calendaryear/'
    urls = []
    for i in range(2013,2021):
        urls.append(strs+str(i))
    start_urls = urls


    def __init__(self,**kwargs):
        self.db = pymysql.connect(host="localhost", port=3306, user="root", passwd='867425', db="weibofilm",charset="utf8")
        # 使用 cursor() 方法创建一个游标对象 cursor
        self.cursor = self.db.cursor()


    @staticmethod
    def close(spider, reason):
        closed = getattr(spider, 'closed', None)
        if callable(closed):
            print('* ' * 50)
            return closed(reason)

    def parse(self, response):
        item = Types()
        totle = response.xpath("..//div[contains(@class,'layer') and contains(@class,'info-layer') and contains(@class,'f12')]")
        up_time = str(response.url).split('/')[-1]
        for i in totle:
            filmname = i.xpath("./div/a/img/@alt").extract()[0]
            temp = i.xpath("./div[2]/p")[-1]
            types = ''

            for j in temp.xpath("./span/a"):
                labes = j.xpath("./text()").extract()[0]
                types = types+labes+' '

            item['filmname'] = filmname
            item['types'] = types
            item['time'] = up_time
            # pdb.set_trace()
            yield item



    # def parse(self,response):
    #
    #     item = Types()
    #     filmname = response.xpath("..//div[@class = 'container-right']/h1/text()").extract()[0].strip() #电影名
    #     up_time = response.xpath("..//div[@class = 'information-list']/span/text()").extract()[0] #上映时间
    #     temp = response.xpath("..//div[@class = 'information-list']/span[@class = 'information-item'][2]/a") #类型
    #     # response.xpath("..//div[@class = 'information-list']/span[@class = 'information-item'][2]").extract()[0]
    #     types = '' #类型
    #     # pdb.set_trace()
    #     for i in temp:
    #         s = i.xpath("./text()").extract()[0] #从当前根目录向下搜索
    #         types = types + s +' '
    #     item['filmname'] = filmname
    #     item['types'] = types
    #     item['time'] = up_time
    #     yield item







