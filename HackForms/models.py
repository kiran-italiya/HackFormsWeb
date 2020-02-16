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
    empty_form = models.CharField(max_length=5000,null=False)
    zip_file = models.CharField(max_length=5000,null=False)
    data = models.CharField(max_length=5000,unique=True,null=False)
    csv_file = models.CharField(max_length=100,null=False)