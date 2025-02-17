from django.urls import path
from . import views

app_name = 'my_classifier'
urlpatterns = [
    path('train/', views.train_model, name='train_model'),
]