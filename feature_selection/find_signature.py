#!/usr/bin/python

import pickle
import numpy
numpy.random.seed(42)


### The words (features) and authors (labels), already largely processed.
### These files should have been created from the previous (Lesson 10)
### mini-project.
words_file = "../text_learning/your_word_data.pkl" 
authors_file = "../text_learning/your_email_authors.pkl"
word_data = pickle.load( open(words_file, "r"))
authors = pickle.load( open(authors_file, "r") )



### test_size is the percentage of events assigned to the test set (the
### remainder go into training)
### feature matrices changed to dense representations for compatibility with
### classifier functions in versions 0.15.2 and earlier
from sklearn import cross_validation
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(word_data, authors, test_size=0.1, random_state=42)

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                             stop_words='english')
features_train = vectorizer.fit_transform(features_train)
features_test  = vectorizer.transform(features_test).toarray()


### a classic way to overfit is to use a small number
### of data points and a large number of features;
### train on only 150 events to put ourselves in this regime
features_train = features_train[:150].toarray()
labels_train   = labels_train[:150]



### your code goes here
from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier()
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)

from sklearn.linear_model import Lasso

#regression = Lasso(alpha=0.2)
#regression.fit(features_train, labels_train)
#coef = regression.coef_.tolist()
#print 'coef_ is a ', coef.__class__.__name__
#max_coef = max(regression.coef_)
#print 'max coef_: ', max_coef
#print 'index: ', coef.index(max_coef)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(labels_test, pred)
print 'accuracy: ', accuracy

importance = clf.feature_importances_.tolist()
max_importance = max(importance)
pos_max_importance = importance.index(max_importance)
print 'max importance: ', max_importance
print 'index of max: ', pos_max_importance

print 'feature word is: ', vectorizer.get_feature_names()[pos_max_importance]

print 'type of importance: ', importance.__class__.__name__

for index, value in enumerate(importance):
    if value > 0.2:
        print 'word: ', vectorizer.get_feature_names()[index], ' has importance of: ', value


