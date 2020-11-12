# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 19:43:51 2020

@author: shahe
"""

#Import necessary libraries
from flask import Flask, render_template, url_for, request
import pandas as pd
import numpy as np
import pickle


#Load the saved model
saved_model = pickle.load(open('spam_clf_model.pkl','rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        mess_vector = tfidf.transform(data).toarray()
        prediction = saved_model.predict(mess_vector)
    return render_template('result.html',prediction = prediction)

if __name__ == '__main__':
    app.run(debug=True)





