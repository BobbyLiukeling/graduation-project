from django.shortcuts import render
from django.views.generic import View
import pdb
import re
# Create your views here.
import os,sys
from getinfo.models import Film  #注意写法
from handleinfo.models import Film_Info

from filmname import filmname

count = 100 #设置获取数据的数量
import pandas as pd



class ResultView(View):
    '''
    结果展示
    '''

    def get(self,request):
        all_film = Film.objects.filter(film_name = filmname)
        all_info = Film_Info.objects.filter(filmname = filmname)[:count]
        pos_index = 0
        neg_index = 0
        high = 0 # 大于0,9的高位评分
        low = 0 # 小于o.1的低位评分
        for i in all_info:
            i = float(i.comment_score)
            if i>0.6:
                pos_index = pos_index+1
            else:
                neg_index = neg_index+1
            if i>0.9:
                high = high+1
            else:
                low = low+1

        sum = neg_index+pos_index
        '''
        正负百分比
        '''
        pos_per = str(round(float(pos_index/(sum)), 4)*100) + '%'
        neg_per = str(round(float(neg_index/(sum)), 4)*100) + '%'

        '''
        高低位
        '''
        high = round(high/sum,4)

        low = round(low/sum)

        comment = ''
        if high>0.25: #高位评分大于25%
            high = str(high * 100)[:2]
            comment = '但是本次爬取的评论中有{}%的评论分数大于0.9（满分为1分）存在大量水军控评现象，拉高评论得分。请谨慎采纳本次推荐'.format(high)
        elif low>0.25:
            low = str(low * 100)[:2]
            comment = '但是本次爬取的评论中有{}%的评论分数小于于0.1（满分为1分）存在大量水军控评现象，拉低评论得分请。谨慎采纳本次推荐'.format(low)
        else:
            comment = '本次评论得分的分布较为均匀,结果较为准确'

        GRANDFA = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

        s1 = all_film[0].pos.split('+') #形成列表
        r1 = r'[\u4e00-\u9fa5]+'
        t1 = []  #积极主题列表
        for i in s1:
            i = re.findall(r1, i)
            for j in i:
                t1.append(j)

        s2 = all_film[0].neg.split('+')
        t2 = [] #消极主题列表
        for i in s2:
            i = re.findall(r1, i)
            for j in i:
                t2.append(j)
        # pdb.set_trace()

        ty = all_film[0].types
        film = all_film[0].film_name
        return render(request,'result.html',{
             'all_film':all_film[0],
            'all_info':all_info,
            'pos' : pos_per,
            'neg' : neg_per,
            'comment':comment,
            't11':t1[0:10],
            't12':t1[10:20],
            't13':t1[20:],
            't21':t2[0:10],
            't22':t2[10:20],
            't23':t2[20:],
            'types':ty,
            'filmname':film

        })


