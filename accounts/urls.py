from django.urls import path

from . import views




urlpatterns = [
    path("signup/", views.SignUpView.as_view(), name="signup"),
    path("change_user_data", views.change_user_data, name="change_user_data"),
    path("is_username_exist", views.is_username_exist, name="is_username_exist"),
    path("is_email_exist", views.is_email_exist, name="is_email_exist"),
]
