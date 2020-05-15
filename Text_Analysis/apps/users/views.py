from django.shortcuts import render
from django.contrib.auth import authenticate,login
from django.http import HttpResponse, HttpResponseRedirect
from django.urls import reverse
from django.views.generic.base import View
# Create your views here.

# def loginView(request):
#     if request.method == 'POST':
#         user_name = request.POST.get('username','')
#         pass_word = request.POST.get('password','')
#         user = authenticate(username=user_name, password=pass_word) #从数据库去除数据并进行验证
#         if user is not None:#验证是否登录
#             login(request, user)
#             '''
#             其中 index 是urls中注册的 name的值
#             path('',TemplateView.as_view(template_name='index.html'),name = 'index'),
#             '''
#             return HttpResponseRedirect(reverse("index"))#登录完成后进入首页
#         else:
#             return render(request, 'login.html', { "msg" : "用户名或密码错误"})
#
#     elif request.method == 'GET':
#         return render(request,'login.html',{})

class loginView(View):
    def get(self, request):
        return render(request, "login.html", {})
    def post(self, request):
        user_name = request.POST.get('username', '')
        pass_word = request.POST.get('password','')
        user = authenticate(username=user_name, password=pass_word) #从数据库去除数据并进行验证
        if user is not None:#验证是否登录
            login(request, user)
            '''
            其中 index 是urls中注册的 name的值
            path('',TemplateView.as_view(template_name='index.html'),name = 'index'),
            '''
            return HttpResponseRedirect(reverse("index"))#登录完成后进入首页
        else:
            return render(request, 'login.html', { "msg" : "用户名或密码错误"})