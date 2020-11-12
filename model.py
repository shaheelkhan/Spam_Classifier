# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 11:36:11 2020

@author: shahe
"""

#Import necessary libraries
import pandas as pd
import numpy as np
import re
import nltk
import pickle
nltk.download('stopwords')
nltk.download('wordnet')

#Import data
messages = pd.read_csv('spam_collection',sep='\t',names = ['label','message'])

#Check the head
messages.head()


#Text Preprocessing
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

wl = WordNetLemmatizer()

def text_preprocess(mess):

  #Remove the punctuations
  new_message = re.sub('[^a-zA-Z]',' ',mess)

  #Lower the cases
  new_message = new_message.lower()

  #Split the words
  new_message = new_message.split()

  #Apply Lemmatization
  new_message = [wl.lemmatize(word) for word in new_message if word not in stopwords.words('english')]

  new_message = ' '.join(new_message)

  return new_message

#Apply the text_preprocessingto the message column
messages['message'] = messages['message'].apply(text_preprocess)


#One hot encode the label
messages['label'] = np.where(messages['label']=='spam',1,0)


#Apply TF-IDF on the text preprocessed data
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

tfidf = TfidfVectorizer()
X = tfidf.fit_transform(messages['message']).toarray()
y = messages['label']

#Save the vectorized message to reuse it again
pickle.dump(tfidf,open('tfidf.pkl','wb'))

#Check the shape of tfidf_message
X.shape

#Train test split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2,random_state=99)


#train the model using Naive Bayes Classifier
from sklearn.naive_bayes import MultinomialNB,GaussianNB
spam_model = GaussianNB()
spam_model.fit(X_train,y_train)

spam_model.score(X_train,y_train)
spam_model.score(X_test,y_test)

#Save the the Model for later use
file = 'spam_clf_model.pkl'
pickle.dump(spam_model,open(file,'wb'))


