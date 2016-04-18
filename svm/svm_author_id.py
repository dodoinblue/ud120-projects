#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###

#########################################################
from sklearn.svm import SVC

clf = SVC(kernel='rbf', C=10000.0)

from time import time
t0 = time()
#clf.fit(features_train[:len(features_train)/100], labels_train[:len(labels_train)/100])
clf.fit(features_train, labels_train)
print 'time train: ', time()-t0

t1 = time()
pred = clf.predict(features_test)
print 'time predict: ', time()-t1

import numpy as np
l = np.ravel(pred).tolist()
print '1 count: ', l.count(1)
print '0 count: ', l.count(0)

print '9, 25, 49'
print pred[9], "\n", pred[25], "\n", pred[49]
#print pred[:50]
print '10, 26, 50'
print pred[10], "\n", pred[26], "\n", pred[50]



from sklearn.metrics import accuracy_score

accuracy = accuracy_score(labels_test, pred)

print accuracy

import matplotlib.pyplot as plt

#ax = np.arange(1, 51)
#res = np.vstack((ax, pred[:50])).T

#print res

def prettyPicture(clf, X_test, y_test):
    x_min = 0.0; x_max = 1.0
    y_min = 0.0; y_max = 1.0

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, m_max]x[y_min, y_max].
    h = .01  # step size in the mesh
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

    plt.pcolormesh(xx, yy, Z, cmap=pl.cm.seismic)

    # Plot also the test points
    grade_sig = [X_test[ii][0] for ii in range(0, len(X_test)) if y_test[ii]==0]
    bumpy_sig = [X_test[ii][1] for ii in range(0, len(X_test)) if y_test[ii]==0]
    grade_bkg = [X_test[ii][0] for ii in range(0, len(X_test)) if y_test[ii]==1]
    bumpy_bkg = [X_test[ii][1] for ii in range(0, len(X_test)) if y_test[ii]==1]

    plt.scatter(grade_sig, bumpy_sig, color = "b", label="fast")
    plt.scatter(grade_bkg, bumpy_bkg, color = "r", label="slow")
    plt.legend()
    plt.xlabel("bumpiness")
    plt.ylabel("grade")
    return plt

#p = prettyPicture(clf, features_test, labels_test)
#p.show()
