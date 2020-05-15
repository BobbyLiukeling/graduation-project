# _*_ encoding:utf-8 _*_
from __future__ import unicode_literals
from datetime import datetime

from django.db import models
from django.contrib.auth.models import AbstractUser



class UserProfile(AbstractUser):
    name = models.CharField(max_length=50, verbose_name=u"用户名")
    # password = models.DateField(verbose_name=u"密码",max_length=100)
    class Meta:
        verbose_name = "用户信息"
        verbose_name_plural = verbose_name

    def __unicode__(self):
        return self.username
