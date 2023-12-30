import string
import joblib
import numpy as np
import pandas as pd
import itertools
import os
from io import BytesIO
import base64
from indoNLP.preprocessing import replace_word_elongation, replace_slang
from nlp_id.lemmatizer import Lemmatizer
from nlp_id.stopword import StopWord 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

class AnalysisSentimentModel:
    vector_text_shape = (4501, 1, 5421)

    def normalize_text(self, text):
        text = text.lower()
        text = text.translate(str.maketrans("","",string.punctuation))
        text = replace_word_elongation(text)
        text = replace_slang(text)
        lemmatizer = Lemmatizer()
        text = lemmatizer.lemmatize(text)
        stopword = StopWord() 
        text = stopword.remove_stopword(text)
        return text
    
    def tf_idf_transform(self, text, tfidf_dir):
        tfidf_vectorizer = TfidfVectorizer()
        tfidf_vectorizer = joblib.load(tfidf_dir)
        tfidf_matrix = tfidf_vectorizer.transform([text])

        # Convert TF-IDF matrix to an array and reshape for LSTM input
        tfidf_matrix_toarray = tfidf_matrix.toarray().reshape(tfidf_matrix.shape[0], 1, tfidf_matrix.shape[1])
        self.vector_text_shape = tfidf_matrix_toarray.shape
        return tfidf_matrix_toarray
    
    def text_preprocessing(self, text, tfidf_dir):
        text = self.normalize_text(text)
        vector_text = self.tf_idf_transform(text, tfidf_dir)
        return vector_text
    
    def get_loaded_model(self, lstm_dir):
        # Define and compile the model
        model = load_model(lstm_dir)
        return model
    
    def predict(self, text, tfidf_dir, lstm_dir):
        vector_text = self.text_preprocessing(text, tfidf_dir)
        model = self.get_loaded_model(lstm_dir)
        prediction = model.predict(vector_text, batch_size=1, verbose=0)
        result = prediction[0][0]
        return result
    
    def create_tf_idf(self, texts, name):
        tfidf_vectorizer = TfidfVectorizer()
        tfidf_matrix = tfidf_vectorizer.fit_transform(texts)

        # Convert TF-IDF matrix to an array and reshape for LSTM input
        tfidf_matrix_toarray = tfidf_matrix.toarray().reshape(tfidf_matrix.shape[0], 1, tfidf_matrix.shape[1])
        joblib.dump(tfidf_vectorizer, "tfidf_"+ name +".pkl")
        self.vector_text_shape = tfidf_matrix_toarray.shape
        return tfidf_matrix_toarray
    
    def create_model(self, lstm_unit, is_regularizer, dropout, recurrent_dropout):
        model = Sequential()
        if is_regularizer:
            model.add(LSTM(
                lstm_unit, 
                input_shape=(
                    self.vector_text_shape[1], 
                    self.vector_text_shape[2]), 
                kernel_regularizer='l2',
                dropout = dropout,
                recurrent_dropout = recurrent_dropout
                ))
        else:
            model.add(LSTM(
                lstm_unit, 
                input_shape=(
                    self.vector_text_shape[1],
                    self.vector_text_shape[2]),
                dropout = dropout,
                recurrent_dropout = recurrent_dropout
                ))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def train_model(self, name, dataset, lstm_unit, is_regularizer, dropout, recurrent_dropout, batch_size, epoch, reduce_lr_patience, early_stop_patience):
        df = pd.read_json(dataset)
        X = df['processed_review'].values
        y = df['sentiment'].values
        X_vectorize = self.create_tf_idf(X, name)

        X_train, X_test, y_train, y_test = train_test_split(X_vectorize, y, test_size=0.2, random_state=42)
        model = self.create_model(lstm_unit=lstm_unit, is_regularizer=is_regularizer, dropout=dropout, recurrent_dropout=recurrent_dropout)
        checkpoint = ModelCheckpoint(
            "lstm_"+ name +".h5", 
            monitor='val_accuracy', 
            verbose=0,
            save_best_only=True, 
            mode='max',
        )

        # Define callbacks for dynamic learning rate and early stopping
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=reduce_lr_patience, min_lr=0.00001)
        early_stop = EarlyStopping(monitor='val_loss', patience=early_stop_patience, restore_best_weights=True)
        callbacks_list = [reduce_lr, early_stop, checkpoint]
        history = model.fit(X_train, y_train, 
                    batch_size=batch_size,
                    epochs=epoch,
                    verbose=0,
                    validation_data=(X_test, y_test),
                    callbacks=callbacks_list
                   )
        
        metrics_df = pd.DataFrame(history.history)
        graph_loss = self.graph_loss(metrics_df)
        graph_acc = self.graph_accuracy(metrics_df)
        model.load_weights("lstm_"+ name +".h5")
        y_pred = model.predict(X_test, batch_size=batch_size)
        y_classes = np.argmax(y_pred,axis=1)
        for i in range(len(y_pred)):
            if(y_pred[i][0]>0.5):
                y_classes[i] = 1
            else:
                y_classes[i] = 0
        y_pred = y_pred[:, 0]
        cm_plot_labels = ['Negative', 'Positive']
        report = pd.DataFrame(
            classification_report(y_test, y_classes, target_names=cm_plot_labels, output_dict=True)
        ).transpose()
        cm = self.plot_confusion_matrix(confusion_matrix(y_test, y_classes), cm_plot_labels, title='Confusion Matrix')
        accuracy = accuracy_score(y_classes, y_test)
        return metrics_df, graph_acc, graph_loss, accuracy, report, cm
    
    def graph_accuracy(self, metrics_df):

        acc = metrics_df['accuracy']
        val_acc = metrics_df['val_accuracy']
        epochs = range(1, len(acc) + 1)
        fig = plt.figure()
        plt.plot(epochs, acc, 'bo', label='Training cat acc')
        plt.plot(epochs, val_acc, 'b', label='Validation cat acc')
        plt.title('Training and validation cat accuracy')
        plt.legend()
        plt.tight_layout()

        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()

        graphic = base64.b64encode(image_png)
        graphic = graphic.decode('utf-8')
        return graphic
    
    def graph_loss(self, metrics_df):

        loss = metrics_df['loss']
        val_loss = metrics_df['val_loss']
        epochs = range(1, len(loss) + 1)
        fig = plt.figure()
        plt.plot(epochs, loss, 'bo', label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title('Training and validation loss')
        plt.legend()
        plt.tight_layout()

        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()

        graphic = base64.b64encode(image_png)
        graphic = graphic.decode('utf-8')
        return graphic
    
    def plot_confusion_matrix(self, cm, classes,
                            normalize=False,
                            title='Confusion matrix',
                            cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        fig = plt.figure()
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")

        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()

        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()

        graphic = base64.b64encode(image_png)
        graphic = graphic.decode('utf-8')
        return graphic
    
    def sentiment_pie_char(self, positive, negative):
        # Data for the pie chart
        labels = ['Sentimen Positif', 'Sentimen Negatif']
        sizes = [positive, negative]

        # Create a pie chart
        fig = plt.figure(figsize=(8, 8))  # Set the size of the pie chart
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle

        # Add a title
        plt.title('Perbandingan Prediksi')

        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()

        graphic = base64.b64encode(image_png)
        graphic = graphic.decode('utf-8')
        return graphic