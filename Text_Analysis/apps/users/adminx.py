# from django.contrib import admin

import xadmin
# Register your models here.
from .models import UserProfile
from xadmin import views

class UserProfileAdmin(object):
    list_display = ['id', 'username','date_joined']
    # pass

class BaseSetting(object): #自定义主题
    enable_themes = True
    use_bootswatch = True

class GlobalSettings(object): #页脚和页底自定义
    site_title = "微博情感分析"
    site_footer = "weibo text sentiment analysis"

xadmin.site.unregister(UserProfile)
# xadmin.site.register(UserProfile, UserProfileAdmin)
xadmin.site.register(UserProfile, UserProfileAdmin)
xadmin.site.register(views.BaseAdminView, BaseSetting)
xadmin.site.register(views.CommAdminView, GlobalSettings)