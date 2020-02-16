from builtins import zip
from zipfile import ZipFile
import os
from django.shortcuts import render
# from django.views.generic import TemplateView
from django.core.files.storage import FileSystemStorage
from HackFormsWeb import settings
from django.shortcuts import render,get_object_or_404
from django.views.generic import UpdateView, ListView

# Create your views here.
from django.http import HttpResponse
from .forms import NewProjectForm
from .models import *

def home(request):
    projects = Project.objects.all()
    return render(request, 'home.html', {'projects':projects})

def new_project(request):
    form = NewProjectForm()
    if(form.is_valid()):
        project = form.save(commit=False)
    else:
        form = NewProjectForm()
    return render(request, 'new_project.html', {'form':form})


class TopicListView(ListView):
    model = Project
    context_object_name = 'project'
    template_name = 'project.html'

    def get_context_data(self, **kwargs):
        # kwargs['project'] = self.project
        return super().get_context_data(**kwargs)

def upload(request):
    if request.method=='POST':
        empty_form = request.FILES['empty_form']
        zip_file = request.FILES['zip_file']
        project_name = request.POST.get('project_name')

        try:
            os.mkdir(os.path.join(settings.MEDIA_ROOT, os.path.join(project_name,'/data')))

        except:pass
        #
        with ZipFile(zip_file) as zip_file:
            names = zip_file.namelist()
            zip_file.extractall(os.path.join(settings.MEDIA_ROOT,project_name+'/data'),names)
        fs = FileSystemStorage(location=os.path.join(settings.MEDIA_ROOT,project_name))
        fs.save(empty_form.name, empty_form)
    return render(request,'home.html',{'project_name':project_name})

def get_queryset(self):
    self.project = get_object_or_404(Project, pk=self.kwargs.get('pk'))
    queryset = self.project
    return queryset
