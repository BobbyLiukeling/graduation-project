# -*- encoding: utf-8 -*-
# @author : bobby
# @time : 2020/2/20 16:15


import xadmin
from .models import Film


class FilmAdmin(object):
    list_display = ['film_name','scrapy_totle','precision_rate','recommend']

xadmin.site.register(Film, FilmAdmin)
