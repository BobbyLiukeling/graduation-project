# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html


import copy
from twisted.enterprise import adbapi
import pymysql
import pymysql.cursors
import pdb
from .items import WeiboItem,CommentItem,Types


class WeiboPipeline(object):
    def __init__(self, dbpool):
        self.dbpool = dbpool

    @classmethod
    def from_settings(cls, settings):
        dbparms = dict(
            host="127.0.0.1",
            port=3306,
            user="root",
            passwd="867425",
            db="weibofilm",
            charset="utf8",
            cursorclass=pymysql.cursors.DictCursor,
            use_unicode=True,
        )
        dbpool = adbapi.ConnectionPool("pymysql", **dbparms)
        return cls(dbpool)

    def process_item(self, item, spider):
        #使用twisted将mysql插入变成异步执行
        query = self.dbpool.runInteraction(self.do_insert, item)
        query.addErrback(self.handle_error, item, spider) #处理异常

    def handle_error(self, failure, item, spider):
        # 处理异步插入的异常
        print(failure)
        print('@'*30)

    def process_item(self, item, spider):
        # 对象拷贝   深拷贝
        asynItem = copy.deepcopy(item)  # 需要导入import copy

        query = self.dbpool.runInteraction(self.do_insert, asynItem)
        query.addErrback(self.handle_error)
        return item

    def do_insert(self, cursor, item):
        if isinstance(item, WeiboItem):
            # 'INSERT INTO tablename(field1,field2, field3, ...) VALUES(value1, value2, value3, ...) ON DUPLICATE KEY UPDATE field1=value1,field2=value2, field3=value3, ...;'
            try:
                sql = "insert into weibo_essay(label,user,content,fav_nums,comment_nums,forward_nums,add_time) values(%s,%s,%s,%s,%s,%s,%s)"\
                      "ON DUPLICATE KEY UPDATE user = {},fav_nums = {}".format(item['user'],item['fav_nums'])
                params = (item["label"], item["user"],item['content'],int(item['fav_nums']),int(item['comment_nums']),int(item['forward_nums']),item['add_time'])
                cursor.execute(sql, params)
            except Exception as e:
                sql = "insert into weibo_essay(label,user,content,fav_nums,comment_nums,forward_nums,add_time) values(%s,%s,%s,%s,%s,%s,%s)"
                      # "ON DUPLICATE KEY UPDATE user = {},fav_nums = {}".format(item['user'], item['fav_nums'])
                params = (
                item["label"], item["user"], item['content'], int(item['fav_nums']), int(item['comment_nums']),
                int(item['forward_nums']), item['add_time']
                )
                cursor.execute(sql, params)

        elif isinstance(item, CommentItem):
            try:
                sql = 'select id from comment where comment_user = "{0}" and essay_id = {1}'.format(item["user"], int(item["essay_id"]))
                cursor.execute(sql)
                if cursor.fetchone() != None: # 查重
                    pass
                else:
                    sql = "insert into comment(essay_id,comment_user,label,fav_nums,add_time,comment_content,score,content_type) values(%s,%s,%s,%s,%s,%s,%s,%s)"
                    params = (int(item["essay_id"]), item["user"], item['label'], int(item['fav_nums']), item['add_time'],item['content'],item['score'],item['type'])
                    # pdb.set_trace()
                    cursor.execute(sql, params)
            except Exception as e:
                print(e)
                pass
        elif isinstance(item, Types):

            try:
                sql = "insert into film_type(filmname,types,times) values(%s,%s,%s)"
                params = (item['filmname'] , item['types'], item['time'])
                # pdb.set_trace()
                cursor.execute(sql, params)
            except Exception as e:
                print(e)
                pass