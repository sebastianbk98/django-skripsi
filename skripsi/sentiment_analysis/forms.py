from django import forms
from .models import Sentiment, LSTMModel

class SentimentForm(forms.ModelForm):
    class Meta:
        model = Sentiment
        fields = ["text"]

class LSTMModelForm(forms.ModelForm):
    class Meta:
        model = LSTMModel
        fields = ["name","dataset", "tfidf", "lstm", "accuracy", "is_selected"]

class UploadDatasetForm(forms.Form):
    name = forms.CharField(max_length=50)
    file = forms.FileField()

class LSTMParameterForm(forms.Form):
    LSTM_Unit = forms.IntegerField(max_value=256, min_value=1)
    L2_Regulizer = forms.BooleanField(label="L2 Regulizer",required=False)
    Dropout = forms.FloatField(min_value=0.0, max_value=1.0)
    Recurrent_Dropout = forms.FloatField(min_value=0.0, max_value=1.0)
    Batch_Size = forms.IntegerField(min_value=1, max_value=256)
    epoch = forms.IntegerField(min_value=1, max_value=200)
    Reduce_Learning_Rate_Patience = forms.IntegerField(min_value=1, max_value=20)
    Early_Stopping_Patience = forms.IntegerField(min_value=1, max_value=256)