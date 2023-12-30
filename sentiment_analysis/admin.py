from django.contrib import admin

# Register your models here.
from .models import Sentiment, LSTMModel

admin.site.register(Sentiment)
admin.site.register(LSTMModel)