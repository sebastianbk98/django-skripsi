from django.db import models
from django.core.validators import FileExtensionValidator
from django.contrib.auth.models import User
from django.db import transaction

# Create your models here.
class Sentiment(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    text = models.TextField()
    prediction = models.FloatField()
    sentiment = models.IntegerField()

    def __str__(self):
        return "Sentimen: " + ("Positif" if self.sentiment==1 else "Negatif")

class LSTMModel(models.Model):
    name = models.CharField(max_length=50, unique=True)
    accuracy = models.FloatField()
    is_selected = models.BooleanField(default=False)
    dataset = models.FileField(upload_to='dataset/', validators=[FileExtensionValidator(['csv'])])
    tfidf = models.FileField(upload_to='tfidf/', validators=[FileExtensionValidator(['pkl'])]) 
    lstm = models.FileField(upload_to='lstm/', validators=[FileExtensionValidator(['h5'])])

    def save(self, *args, **kwargs):
        if not self.is_selected:
            return super(LSTMModel, self).save(*args, **kwargs)
        with transaction.atomic():
            LSTMModel.objects.filter(
                is_selected=True).update(is_selected=False)
            return super(LSTMModel, self).save(*args, **kwargs)