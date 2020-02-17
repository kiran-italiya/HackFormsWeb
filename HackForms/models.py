from django.db import models


class Project(models.Model):
    project_name = models.CharField(max_length=100, unique=True,primary_key=True)
    empty_form = models.FilePathField(path='/media/')
    zip_file = models.FilePathField(path='/media/')
    data = models.FilePathField(path='/media/')
    csv_file = models.FilePathField(path='/media/')
    def __str__(self):
        return  self.project_name