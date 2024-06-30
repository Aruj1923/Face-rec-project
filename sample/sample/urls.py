"""
URL configuration for sample project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.0/topics/http/urls/
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
from django.urls import path
from home.views import *
# from accounts.views import VideoCamera
from accounts.views import *


urlpatterns = [
    path( 'contact',contact,name= 'contact'),
    path('about/contact',contact,name= 'contact'),
    path('' , home_view , name = 'home'),
    path('about/',  about  , name = 'about'),
    path( 'about/home' , home_view , name = 'home'),
    path('about/account', home, name='home'),
    path('newform',form_page, name = "newform"),
    path('about/newform',form_page, name = "newform"),
    path('home' , home_view , name = 'home'),
    path('about/about' , about,name='about'),
    path('account', home , name = 'home'),
    path('about/form_page', form_page , name = 'form_page'),
    path('video_cap', video_cap , name = 'video_cap'),
    # path('front', front , name = 'front'),
    path('admin/', admin.site.urls),
]
