from django import forms

from .models import Project

class NewProjectForm(forms.ModelForm):

    class Meta:
        model = Project
        fields = ['project_name','empty_form','zip_file']