"""Text_Analysis URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
import xadmin
from django.views.generic import TemplateView
from django.urls import include,re_path,path

from users.views import  loginView
from getinfo.views import GetInfoView
from handleinfo.views import HandleView
from result.views import ResultView


urlpatterns = [
    path('xadmin/', xadmin.site.urls),
    path('',TemplateView.as_view(template_name='index.html'),name = 'index'),
    path('login/', loginView.as_view(), name = 'login'),
    path('getinfo/',GetInfoView.as_view(),name = 'getinfo'),
    path('handleinfo/',HandleView.as_view(),name = 'handleinfo'),
    path('result/',ResultView.as_view(),name = 'result'),
]
