# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html




import scrapy
import datetime
from scrapy.loader.processors import MapCompose


def data_convert(value):
    try:
        create_date = datetime.datetime.strptime(value, "%Y-%m-%d").date()
    except Exception as e:
        create_date = datetime.datetime.now().date()
    return create_date



class WeiboItem(scrapy.Item):
    # define the fields for your item here like:
    # name = scrapy.Field()
    label = scrapy.Field()
    user = scrapy.Field()
    content = scrapy.Field()
    fav_nums = scrapy.Field()
    comment_nums = scrapy.Field()
    forward_nums = scrapy.Field()
    add_time = scrapy.Field(
        input_processor=MapCompose(data_convert),
    )

class CommentItem(scrapy.Item):
    user = scrapy.Field()
    label = scrapy.Field()
    essay_id = scrapy.Field()
    fav_nums = scrapy.Field()
    content = scrapy.Field()
    add_time = scrapy.Field(
        input_processor=MapCompose(data_convert),
    )
    score = scrapy.Field()
    type = scrapy.Field()

class Types(scrapy.Item):
    filmname = scrapy.Field()
    types = scrapy.Field() #电影类型
    time = scrapy.Field() #上映时间

