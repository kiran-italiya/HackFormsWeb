# Generated by Django 2.2.2 on 2020-02-16 06:42

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Project',
            fields=[
                ('project_name', models.CharField(max_length=100, primary_key=True, serialize=False, unique=True)),
                ('empty_form', models.CharField(max_length=5000)),
                ('zip_file', models.CharField(max_length=5000)),
                ('data', models.CharField(max_length=5000, unique=True)),
                ('csv_file', models.CharField(max_length=100)),
            ],
        ),
    ]