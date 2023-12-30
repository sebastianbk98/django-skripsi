from django.urls import reverse_lazy
from django.views import generic
from .forms import SignUpForm, ChangeUserDataForm
from django.shortcuts import render, redirect
from django.contrib.auth.models import User
from django.http import JsonResponse, HttpResponse

class SignUpView(generic.CreateView):
    form_class = SignUpForm
    success_url = reverse_lazy("login")
    template_name = "registration/signup.html"

def change_user_data(request):
    if request.method == "POST":
        request.user.username = request.POST.get("username")
        request.user.email = request.POST.get("email")
        request.user.save()
        return redirect('home')
    data = {
        'username':request.user.username,
        'email':request.user.email}
    form = ChangeUserDataForm(initial=data)
    context ={'form':form}
    return render(request, "registration/change_user_data.html",context)

def is_username_exist(request):
    if request.method == "POST":
        try:
            if User.objects.filter(username=request.POST.get('username')).exists():
                return JsonResponse({'success':False})
            else:
                return JsonResponse({'success':True})
        except User.DoesNotExist:
            return JsonResponse({'success':True})

def is_email_exist(request):
    if request.method == "POST":
        try:
            if User.objects.filter(email=request.POST.get('email')).exists():
                return JsonResponse({'success':False})
            else:
                return JsonResponse({'success':True})
        except User.DoesNotExist:
            return JsonResponse({'success':True})