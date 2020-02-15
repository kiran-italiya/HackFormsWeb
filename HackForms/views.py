from django.http import HttpResponse
from django.shortcuts import render
from templates import *


def home(request):
    return render(request, 'home.html')