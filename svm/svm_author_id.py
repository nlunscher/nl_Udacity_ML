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
from sklearn import svm


clf = svm.SVC(kernel = "rbf", C = 10000, max_iter = 2000)
print "starting fitting"
t0 = time()
#clf.fit(features_train[:len(features_train)/100], labels_train[:len(features_train)/100])
clf.fit(features_train, labels_train)
print "train time:", round(time()-t0,3), "s"

t0 = time()
pred = clf.predict(features_test)
print "test time:", round(time()-t0,3), "s"

acc = clf.score(features_test, labels_test)
print acc

for i in [10,26,50]:
	print pred[i]

print sum(pred)
print len(features_test) - sum(pred)
#########################################################


