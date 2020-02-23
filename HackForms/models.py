from django.db import models
from HackFormsWeb import settings

# class Project(models.Model):
#     project_name = models.CharField(max_length=100, unique=True,primary_key=True)
#     empty_form = models.FilePathField(path='/media/')
#     zip_file = models.FilePathField(path='/media/')
#     data = models.FilePathField(path='/media/')
#     csv_file = models.FilePathField(path='/media/')
#     def __str__(self):
#         return  self.project_name
def user_directory_path(instance, filename):
    return '{0}/{1}'.format(instance.project_name,filename)
class Project(models.Model):
    project_name = models.CharField(max_length=100, unique=True,primary_key=True)
    empty_form = models.FileField(upload_to=user_directory_path)
    zip_file = models.FileField(upload_to=user_directory_path)
    csv_file = models.FileField(upload_to=settings.MEDIA_ROOT)
    def __str__(self):
        return  self.project_name