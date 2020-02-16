from django.db import models

# Create your models here.
# class NewProjectForm(models.Model):
#     def __init__(self,proj_name):
#         self.project_name = proj_name
#     project_name = models.CharField(max_length=50,unique=True)
#     empty_form = models.FileField(upload_to=project_name)
#     zip_file = models.FileField(upload_to=project_name)

class Project(models.Model):
    project_name = models.CharField(max_length=100, unique=True,primary_key=True)
    empty_form = models.FilePathField(null=True)
    zip_file = models.FilePathField(null=True)
    data = models.FilePathField(null=True)
    csv_file = models.FilePathField(null=True)
    def __str__(self):
        return  self.project_name