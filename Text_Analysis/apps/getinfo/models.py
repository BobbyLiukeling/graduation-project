from __future__ import unicode_literals
from datetime import datetime

from django.db import models
from DjangoUeditor.models import UEditorField
# Create your models here.

class Film(models.Model):
    film_name = models.CharField(max_length=20,verbose_name='电影名')
    scrapy_totle = models.IntegerField(default=0,verbose_name='爬取的总条数')
    mean = models.FloatField(default=0,verbose_name='平均数',blank=True)
    pos = models.TextField(verbose_name='正向主题')
    neg = models.TextField(verbose_name='负向主题')

    types = models.CharField(verbose_name='电影类型',blank=True,max_length=200)

    class Meta:
        verbose_name = u"电影详情，及分析结果"
        verbose_name_plural = verbose_name

    def __unicode__(self):
        return self.name







