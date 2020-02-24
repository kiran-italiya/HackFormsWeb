from django.db import models
from HackFormsWeb import settings


class Project(models.Model):
    project_name = models.CharField(max_length=100, unique=True,primary_key=True)
    empty_form = models.FileField(path=settings.MEDIA_ROOT)
    zip_file = models.FileField(path=settings.MEDIA_ROOT)
    data = models.FileField(path=settings.MEDIA_ROOT)
    csv_file = models.FileField(path=settings.MEDIA_ROOT)
    def __str__(self):
        return  self.project_name