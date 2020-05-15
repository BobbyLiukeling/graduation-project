# -*- encoding: utf-8 -*-
# @author : bobby
# @time : 2020/2/20 15:29

import xadmin
from .models import Film_Info


class Film_InfoAdmin(object):
    list_display = ['film_name','comment_user','comment_content','comment_score','sentiment_attent','add_time',]

xadmin.site.register(Film_Info,Film_InfoAdmin)