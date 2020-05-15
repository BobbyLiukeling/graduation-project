from __future__ import unicode_literals
from datetime import datetime
from django.db import models
import time
import datetime as dt
from getinfo.models import Film
from DjangoUeditor.models import UEditorField


# Create your models here.




class Film_Info(models.Model):
    # film_name = models.ForeignKey(Film,verbose_name=u"电影id", on_delete=models.CASCADE)
    add_time = models.DateTimeField(default=datetime.now)
    comment_user = models.CharField(max_length=50, verbose_name='评论用户')
    comment_content = models.CharField(max_length=500, verbose_name='评论内容')
    comment_score = models.FloatField(max_length=50, verbose_name='情感评分')
    filmname = models.CharField(verbose_name=u"电影名",max_length=50,default=True)
    # sentiment_attent = models.IntegerField(max_length=2, verbose_name='情感倾向')  # 正为1 ，负向为-1

    class Meta:
        verbose_name = u"电影分析过程"
        verbose_name_plural = verbose_name

    def __unicode__(self):
        return self.name





