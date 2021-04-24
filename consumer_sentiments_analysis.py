#standard ML libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
 #for importing regular expressions
 
import re

#for reading dataset & delimiter = \t coz it's a tab seperated values

dataset = pd.read_csv('consumer_sentiments.tsv' , delimiter = '\t') 

########################STEPS FOR CLEANING ONE REVIEW#####################

#for first review from dataset

first = dataset['Review'][0] 

#1. Remove all the punctuations, numbers, symbols, emojis and unwanted characters.

text = re.sub('[^a-zA-Z]', ' ', first)

#2. Get all the data in lower case.

text = text.lower()

#for implementing 3rd step we need to convert text (string) into list

text = text.split()

#3. Remove unwanted words like preprositions, conjunctions , determiners, fillers, pronouns etc. 
#we import library called stopwords

from nltk.corpus import stopwords

t1 = [word for word in text if not word in set(stopwords.words('english'))]

#4. Perform stemming or lemmatization.

from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

t2 = [ps.stem(word) for word in text if not word in set(stopwords.words('english'))]

clean_text = ' '.join(t2) 

###################NOW FOR COMPLETE DATASET######################

clean_reviews = []
for i in range(1000):
    text = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    text = text.lower()
    text = text.split()
    # t1 = [word for word in text if not word in set(stopwords.words('english'))]
    text = [ps.stem(word) for word in text if not word in set(stopwords.words('english'))]
    text = ' '.join(text)
    clean_reviews.append(text)

#5. Represent the data using an nlp model.
# model here we use is BAG OF WORDS {BOW}

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 500)

#to convert data into sparse matrix

x = cv.fit_transform(clean_reviews)

#to convert data into array form which is visible to everyone

x = x.toarray()

#y for testing 

y = dataset['Liked'].values

#FOR SPLITTING THE DATASET INTO TRAINING & TESTING DATASET

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y)

#NOW FOR CLASSIFICATION WE USE LOGISTIC REGRESSION ALGORITHM

from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()

#training the model

log_reg.fit(x_train, y_train)

#checkin accuracy of the training dataset

acc = log_reg.score(x_train,y_train)

#for checking what max features names are

feature_name = cv.get_feature_names()

#testing the model

log_reg.fit(x_test, y_test)

#for predictions 

y_pred = log_reg.predict(x_test)

#for checking the accuracy of model

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test , y_pred))

####################new review testing######################################

#taking review from user

review = input('enter your feedback \n')

#applying same steps(1-5) on the inputted review

new_review = []
text = re.sub('[^a-zA-Z]', ' ', review)
text = text.lower()
text = text.split()
text = [ps.stem(word) for word in text if not word in set(stopwords.words('english'))]
text = ' '.join(text)
new_review.append(text)
x_new = cv.transform(new_review)
x_new = x_new.toarray()

#getting the prediction from model

new_pred = log_reg.predict(x_new)

#analysing the prediction

if new_pred[0]==1:
    print("Review is Positive")
else:
    print("Review is Negative")