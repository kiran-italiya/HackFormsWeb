from builtins import zip
from zipfile import ZipFile
import os
from django.shortcuts import render,redirect
# from django.views.generic import TemplateView
from django.core.files.storage import FileSystemStorage
from HackForms.Processing import main as hk
from HackFormsWeb import *
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

    def get_queryset(self):
        self.project = get_object_or_404(Project, pk=self.kwargs.get('pk'))
        queryset = self.project
        return queryset

def upload(request):
    if request.method=='POST':
        form = NewProjectForm(request.POST, request.FILES)
        context={}
        if form.is_valid():
            # form.save(commit=False)
            # context['empty_form']=empty_form
            empty_form = request.FILES['empty_form']
            zip_file = request.FILES['zip_file']
            project_name = request.POST.get('project_name')
            context['project_name']=project_name
            try:
                os.mkdir(os.path.join(settings.MEDIA_ROOT, os.path.join(project_name,'/data')))
            except:pass

                    # form.empty_form = os.path.join(settings.MEDIA_URL,project_name,empty_form.name).replace('\\','/')
                    # print(form.empty_form)
            with ZipFile(zip_file) as zip_file:
                for i,f in enumerate(zip_file.filelist):
                    f.filename = '{0}.jpg'.format(i)
                    zip_file.extract(f,os.path.join(settings.MEDIA_ROOT,project_name+'/data'))
                # names = zip_file.namelist()
                        # names = [i for i,_ in enumerate(zip_file.namelist())]
                # zip_file.extractall(os.path.join(settings.MEDIA_ROOT,project_name+'/data'),names)

        # fs = FileSystemStorage(location=os.path.join(settings.MEDIA_ROOT,project_name))
        # name = fs.save(empty_form.name, empty_form)
        # context['empty_form']=fs.url(name)
            form.save()
            projects = Project.objects.all()
            return redirect(home)
        else:
            form = NewProjectForm()
        return render(request, 'new_project.html', {'form': form})
    else:
        form = NewProjectForm()
    return render(request, 'new_project.html', {'form': form})


def open_project(request,pk):
    project = get_object_or_404(Project, pk=pk)
    return render(request,'project.html',{'project':project})
#
def generate(request,pk):
    project = get_object_or_404(Project,pk=pk)
    pf = hk.ProcessForm()
    df_final = pf.processForm(os.path.join(settings.BASE_DIR,'HackForms/Processing/k4.jpg') ,os.path.join(settings.BASE_DIR,"HackForms/Processing/data/k4/"))
    df_final.to_csv('media/'+project.project_name+'/data.csv')
    project.csv_file=project.project_name+'/data.csv'
    project.save()
    return render(request, 'project.html', {'project': project})