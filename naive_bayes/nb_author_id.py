#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project. 

    Use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess

#change

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###

print "lengths:", len(features_train), len(features_test), len(labels_train), len(labels_test)
print  len(features_train[0])
print features_train[1:10]
print sum(features_train[0])

# for e in features_train[0]:
# 	print e


from sklearn.naive_bayes import GaussianNB

clf = GaussianNB()

t0 = time()
clf.fit(features_train, labels_train)
print "train time:", round(time()-t0,3), "s"

t0 = time()
pred = clf.predict(features_test)
print "test time:", round(time()-t0,3), "s"

acc = clf.score(features_test, labels_test)
print acc

#########################################################


