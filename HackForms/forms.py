from django import forms

from .models import Project

class NewProjectForm(forms.ModelForm):
    project_name = forms.CharField(
        widget=forms.TextInput(
            attrs={'placeholder': 'Project Name'}
        ),
        max_length=4000,
        help_text='Enter a unique name'
    )
    empty_form = forms.FileField()
    zip_file = forms.FileField()

    class Meta:
        model = Project
        fields = ['project_name','empty_form','zip_file']