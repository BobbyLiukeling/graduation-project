
from django.shortcuts import render
from django.views.generic import View
from scrapy.cmdline import execute
import os,sys
from scrapyd_api import ScrapydAPI
import pdb
import time


class GetInfoView(View):
    '''
    开始爬取，获取信息
    '''

    def get(self,request):
        return render(request,'get-info.html',{})

    def post(self,request):
        '''
        将前端传入的电影名存入爬虫文件夹下，等待下一步进行爬取
        '''

        filmname = request.POST.get('filmname')
        filmname = 'filmname = "'+filmname+'"'+'\n # 从前端传入Django的电影名，存入爬虫文件夹等待下一步进行爬取电影'
         # file = os.path.abspath(os.path.join(os.getcwd(), "../..")) # OS获取的是manage.py的上上层目录
        # file = os.path.abspath(os.path.join(os.getcwd(), ".."))  # OS获取的是manage.py的上层目录 Django
        # a = open('weibo/weibo/spiders/filmname.py','w',encoding='utf-8')
        a = open('gerapy/projects/weibo/weibo/spiders/filmname.py','w',encoding='utf-8') #路径是相对于manage.py文件,爬虫获取电影名
        a.write(filmname)
        a.close()
        b= open('../Data_management/filmname.py', 'w', encoding='utf-8')  # 路径是相对于manage.py文件，文件处理获取电影名
        b.write(filmname)
        b.close()
        c = open('apps/filmname.py', 'w', encoding='utf-8')  # 路径是相对于manage.py文件，文件处理获取电影名
        c.write(filmname)
        c.close()

        # # 加入cwd 并切换工作目录
        # GRANDFA = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
        # os.chdir(GRANDFA+'/Data_Management') #跳转到待执行目录的父目录下
        # # pdb.set_trace()
        # # os.system('cd Data_Management')
        # os.system('python data_handle.py') #对数据进行处理

        return render(request,'get-info.html',{})

