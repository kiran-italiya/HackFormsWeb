from django.contrib import admin
from django.urls import path

from HackForms import views

from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('', views.home, name='home'),
    path('new_project/', views.new_project, name='new_project'),
    path('project/<str:pk>', views.open_project, name='projects'),
    path('new_project/upload/', views.upload, name='upload'),
    path('admin/', admin.site.urls),
    path('project/<str:pk>/generate_analytics',views.generate,name='generate')
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
