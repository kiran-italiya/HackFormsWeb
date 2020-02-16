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