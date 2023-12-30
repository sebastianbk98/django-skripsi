import json 
import pandas as pd
import os
from django.http import FileResponse
from django.views import View
from django.shortcuts import render, redirect
from django.http import JsonResponse, HttpResponse
from django.urls import reverse_lazy
from .models import Sentiment, LSTMModel
from django.contrib.auth.models import User
from .forms import SentimentForm, LSTMModelForm, UploadDatasetForm, LSTMParameterForm
from django.contrib.admin.views.decorators import staff_member_required
from django.contrib.auth.decorators import login_required
from .prediction.lstm import AnalysisSentimentModel
import requests

# Create your views here.
@staff_member_required(login_url=reverse_lazy('admin:login'))
def admin_index(request):
    users_count = User.objects.count()
    predictions_count = Sentiment.objects.count()
    lstm_model_count = LSTMModel.objects.count()
    negative_sentiment = Sentiment.objects.filter(sentiment=0).count()
    positive_sentiment = Sentiment.objects.filter(sentiment=1).count()
    context = {
        'users':users_count,
        'predictions':predictions_count,
        'lstm_model':lstm_model_count
    }
    if not (negative_sentiment == 0 and positive_sentiment == 0):
        obj = AnalysisSentimentModel()
        pie_chart = obj.sentiment_pie_char(positive_sentiment, negative_sentiment)
        context['pie_chart'] = pie_chart
    
    
    return render(request, 'sentiment_analysis/admin/index.html', context)

@staff_member_required(login_url=reverse_lazy('admin:login'))
def model_list(request):
    form = LSTMModelForm()
    models = LSTMModel.objects.all()
    selected_model = LSTMModel.objects.filter(is_selected=True).first()
    context = {
        'models':models,
        'selected_model' : selected_model,
        'form':form
    }
    
    if request.method == 'POST':
        form = LSTMModelForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            form = LSTMModelForm()
            context = {
                'models':models,
                'selected_model' : selected_model,
                'form':form
            }
            return render(request, 'sentiment_analysis/admin/models_list.html', context)
    return render(request, 'sentiment_analysis/admin/models_list.html', context)
    

@login_required(login_url=reverse_lazy('login'))
def dashboard(request):
    last5predictions = Sentiment.objects.filter(user=request.user).order_by('-id')[:5]
    model_query = LSTMModel.objects.filter(is_selected=True)
    if request.method == 'POST':
        form = SentimentForm(request.POST)
        if form.is_valid():
            lstm_model = AnalysisSentimentModel()
            active_model = model_query.first()
            tfidf_dir = active_model.tfidf.path
            lstm_dir = active_model.lstm.path
            data = form.save(commit=False)
            data.user = request.user
            data.prediction = lstm_model.predict(request.POST.get('text'), tfidf_dir, lstm_dir)
            data.sentiment = 1 if data.prediction>0.5 else 0
            data.admin = 1
            data.user_sentiment = 1
            data.save()
            context = {
                'prediction':data,
                'last5predictions':last5predictions,
                'form':SentimentForm()
                }
            return render(request, 'sentiment_analysis/prediction.html', context)
    context = {
        'last5predictions':last5predictions,
        'form':SentimentForm()
        }
    if model_query.count() == 0:
        context['no_model'] = True
    return render(request, 'sentiment_analysis/prediction.html', context)

@staff_member_required(login_url=reverse_lazy('admin:login'))
def upload_dataset(request):
    session_clear(request)
    if request.method == "POST":
        form = UploadDatasetForm(request.POST, request.FILES)
        if form.is_valid():
            df = pd.read_csv(request.FILES["file"])
            df = df[["content", "sentiment"]]
            request.session['dataset'] = df.to_json()
            request.session['name'] = request.POST.get('name')
            return redirect('normalize-dataset')
    else:
        form = UploadDatasetForm()
    return render(request, "sentiment_analysis/admin/upload_dataset.html", {"form": form})

@staff_member_required(login_url=reverse_lazy('admin:login'))
def normalize_dataset(request):
    if request.session.get('name') is None:
        return redirect('upload-dataset')
    df = pd.read_json(request.session.get('dataset'))
    data = df_to_table(df.head())
    if request.method == "POST":
        lstm_model = AnalysisSentimentModel()
        df['processed_review'] = df['content'].apply(lambda x: lstm_model.normalize_text(x))
        request.session['dataset'] = df.to_json()
        return redirect('train-model')
    context = {'data':data}
    return render(request, "sentiment_analysis/admin/normalize_dataset.html", context)

@staff_member_required(login_url=reverse_lazy('admin:login'))
def train_model(request):
    if request.session.get('name') is None:
        return redirect('upload-dataset')
    if 'processed_review' not in pd.read_json(request.session.get('dataset')).columns:
        return redirect('normalize-dataset')
    if request.method == "POST":
        form = LSTMParameterForm(request.POST)
        if form.is_valid():
            lstm_model = AnalysisSentimentModel()
            name=request.session.get('name')
            metric_df, graph_acc, graph_loss, accuracy, report, cm = lstm_model.train_model(
                name=name,
                dataset=request.session.get('dataset'),
                lstm_unit=int(request.POST.get('LSTM_Unit')),
                is_regularizer=request.POST.get('L2_Regulizer'),
                dropout=float(request.POST.get('Dropout')),
                recurrent_dropout=float(request.POST.get('Recurrent_Dropout')),
                batch_size=int(request.POST.get('Batch_Size')),
                reduce_lr_patience=int(request.POST.get('Reduce_Learning_Rate_Patience')),
                early_stop_patience=int(request.POST.get('Early_Stopping_Patience')),
                epoch=int(request.POST.get('epoch')))
            
            history_data = df_to_table(metric_df)
            report = report.rename({'f1-score': 'f1_score'}, axis='columns')
            request.session['report'] = report.to_json()
            request.session['cm'] = cm
            request.session['acc'] = accuracy
            
            context = {
                'form': form,
                'history_data':history_data,
                'graph_loss':graph_loss,
                'graph_acc':graph_acc,
                }
            return render(request, 'sentiment_analysis/admin/train_model.html', context)
    else:
        df = pd.read_json(request.session['dataset'])
        data = df_to_table(df.head())
        form = LSTMParameterForm()
        context = {'data': data, 'form': form}
    return render(request, "sentiment_analysis/admin/train_model.html", context)

@staff_member_required(login_url=reverse_lazy('admin:login'))
def evaluation(request):
    if request.session.get('name') is None:
        return redirect('upload-dataset')
    if 'processed_review' not in pd.read_json(request.session.get('dataset')).columns:
        return redirect('normalize-dataset')
    if request.session.get('acc') is None:
        return redirect('train-model')
    name = request.session.get('name')
    df = pd.read_json(request.session.get('dataset'))
    data = df_to_table(df.head())
    df[["content", "sentiment"]].to_csv("dataset_"+name+".csv")
    report = pd.read_json(request.session.get('report'))
    report = df_to_table(report)
    cm = request.session['cm']
    
    keras_model = LSTMModel(name=name)
    keras_model.accuracy = float(request.session.get('acc'))
    keras_model.tfidf.save(
        name +".pkl", 
        open("tfidf_"+ name +".pkl", 'rb'))
    keras_model.lstm.save(
        name +".h5", 
        open("lstm_"+ name +".h5", 'rb'))
    keras_model.dataset.save(
        name+".csv", 
        open("dataset_"+name+".csv", 'rb'))
    keras_model.save()

    if os.path.exists("tfidf_"+ name +".pkl"):
        os.remove("tfidf_"+ name +".pkl")
    if os.path.exists("lstm_"+ name +".h5"):
        os.remove("lstm_"+ name +".h5")
    if os.path.exists("dataset_"+name+".csv"):
        os.remove("dataset_"+name+".csv")
    
    context = {
        'data': data,
        'report' : report,
        'cm':cm
    }
    session_clear(request)
    return render(request, "sentiment_analysis/admin/evaluation.html", context)

def session_clear(request):
    if request.session.get('name') is not None:
        del request.session['name']
    if request.session.get('dataset') is not None:
        del request.session['dataset']
    if request.session.get('acc') is not None:
        del request.session['acc']
    if request.session.get('report') is not None:
        del request.session['report']
    if request.session.get('cm') is not None:
        del request.session['cm']

def df_to_table(dataframe):
    json_records = dataframe.reset_index().to_json(orient ='records') 
    data = [] 
    data = json.loads(json_records)
    return data

def form_model_validation(request):
    if request.method == "POST":
        form = LSTMParameterForm(request.POST)
        if form.is_valid():
            return JsonResponse({'success':True})
        else:
            return JsonResponse({'success':False})
    pass

def is_name_exist(request):
    if request.method == "POST":
        try:
            if LSTMModel.objects.filter(name=request.POST.get('name')).exists():
                return JsonResponse({'success':False})
            else:
                return JsonResponse({'success':True})
        except LSTMModel.DoesNotExist:
            return JsonResponse({'success':True})

def get_model(request):
    if request.method == "POST":
        model = list(LSTMModel.objects.filter(name=request.POST.get('name')).values())
        return JsonResponse({'model':model})

def set_model_active(request):
    if request.method == "POST":
        model = LSTMModel.objects.filter(name=request.POST.get('name')).first()
        model.is_selected = True
        model.save()
        return JsonResponse({'success':True})


class DownloadFileView(View):
    def get(self, request):
        try:
            # Retrieve the YourModel object using primary key (pk)
            name = request.GET.get('name')
            type = request.GET.get('type')
            download_model = LSTMModel.objects.filter(name=name).first()
            if type == 'dataset':
                file = download_model.dataset
            elif type == 'tfidf':
                file = download_model.tfidf
            elif type == 'lstm':
                file = download_model.lstm
            else:
                return HttpResponse("Url not found")
            if file:
                # Open the file associated with model_file field and serve it as a FileResponse
                response = FileResponse(file)
                response['Content-Disposition'] = 'attachment; filename="%s"' % file.name
                return response
            else:
                return HttpResponse("File not found")
        except download_model.DoesNotExist:
            return HttpResponse("Model not found")
