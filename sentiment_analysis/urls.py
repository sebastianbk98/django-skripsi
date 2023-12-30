from django.urls import path
from . import views


urlpatterns = [
    # path("signup/", SignUpView.as_view(), name="signup"),
    path("", views.dashboard, name="analysis-dashboard"),
    path("admin/", views.admin_index, name="admin-index"),
    path("admin/model-list", views.model_list, name="model-list"),
    path("admin/get-model", views.get_model, name="get-model"),
    path("admin/set-model-active", views.set_model_active, name="set-model-active"),
    path("admin/upload-dataset", views.upload_dataset, name="upload-dataset"),
    path("admin/is_name_exist", views.is_name_exist, name="is_name_exist"),
    path("admin/normalize-dataset", views.normalize_dataset, name="normalize-dataset"),
    path("admin/train-model", views.train_model, name="train-model"),
    path("admin/train-model/validation", views.form_model_validation, name="form_model_validation"),
    path("admin/evaluation", views.evaluation, name="evaluation"),
    path("admin/model-list/download", views.DownloadFileView.as_view(), name="download-file"),
]
