from django.shortcuts import render
from django.views.generic import View
import os, sys
import pdb

from getinfo.models import Film  #注意写法
from .models import Film_Info
# import ..filmname.film as film#默认apps为根目录


from filmname import filmname
count = 100 #设置获取数据的数量

class HandleView(View):
    '''
    信息处理
    '''

    def get(self,request):
        # 加入cwd 并切换工作目录
        GRANDFA = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
        os.chdir(GRANDFA + '/Data_Management')  # 跳转到待执行目录的父目录下
        # pdb.set_trace()
        # os.system('cd Data_Management')
        os.system('python data_handle.py')  # 对数据进行处理


        all_film = Film.objects.filter(film_name = filmname)
        s = len(Film_Info.objects.filter(filmname=filmname))
        all_info = Film_Info.objects.filter(filmname=filmname)[:count]

        scrapy_totle = all_film[0].scrapy_totle
        return render(request,'handle-info.html',{
            'all_film':all_film,
            'all_info':all_info,
            'count' : s,
            'scrapy_totle':scrapy_totle,
        })

        # return render(request, 'handle-info.html', {})

    # def post(self,request):
    #     return render(request,'handle-info.html',{})