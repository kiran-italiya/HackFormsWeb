from builtins import zip
from zipfile import ZipFile
import os
from django.shortcuts import render
# from django.views.generic import TemplateView
from django.core.files.storage import FileSystemStorage
from HackFormsWeb import settings
# Create your views here.
from django.http import HttpResponse


def home(request):
    
    return render(request, 'home.html')

def upload(request):
    if request.method=='POST':
        empty_form = request.FILES['empty_form']
        zip_file = request.FILES['zip_file']
        fs = FileSystemStorage()
        fs.save(empty_form.name,empty_form)
        try:
            os.mkdir(os.path.join(settings.MEDIA_ROOT, 'data'))
        except:pass
        with ZipFile(zip_file) as zip_file:
            names = zip_file.namelist()
            zip_file.extractall(os.path.join(settings.MEDIA_ROOT,'data'),names)
    return render(request,'home.html')