from django.contrib.auth import views as auth_views
from django.urls import path

from . import views

urlpatterns = [
    path(
        "login/",
        auth_views.LoginView.as_view(template_name="users/login.html"),
        name="login",
    ),
    path("logout/", auth_views.LogoutView.as_view(next_page="home"), name="logout"),
    path("register/", views.register_view, name="register"),
    path("profile/", views.profile_view, name="profile"),
    path("password_change/",
         auth_views.PasswordChangeView.as_view(template_name="users/password_change.html"),
         name="password_change"),
    path("password_change/done/",
         auth_views.PasswordChangeDoneView.as_view(template_name="users/password_change_done.html"),
         name="password_change_done"),
]
