from django.contrib import admin
from django.urls import path
from GenuineApp import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.index, name='index'),
    path('register/', views.Register, name='register'),
    path('register/action/', views.RegisterAction, name='register_action'),
    path('login/', views.UserLogin, name='login'),
    path('login/action/', views.UserLoginAction, name='login_action'),
    path('logout/', views.UserLogout, name='logout'),
    path('dashboard/', views.Dashboard, name='dashboard'),
    path('load-dataset/', views.LoadDataset, name='load_dataset'),
    path('train-model/', views.TrainModel, name='train_model'),
    path('predict/', views.Predict, name='predict'),
    path('predict/text/', views.PredictTextAction, name='predict_text'),
    path('predict/url/', views.PredictURLAction, name='predict_url'),
    path('predict/file/', views.PredictFileAction, name='predict_file'),
    path('results/', views.Results, name='results'),
]