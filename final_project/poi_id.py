#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary',
				# 'total_payments',
				'bonus',
				'total_stock_value',
				'from_this_person_to_poi',
				'from_poi_to_this_person'
				] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

# print my_dataset["GLISAN JR BEN F"]

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.

from sklearn.model_selection import GridSearchCV
scoring_options = ["accuracy","recall", "precision", "f1"]

tree_parameters = {"criterion":("gini", "entropy"), 
				"min_samples_split":(range(2,30,2)),
				# "splitter":("best", "random"),
				"min_samples_leaf":(range(1,5))
				}

svc_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

rfc_parameters = {"criterion":("gini", "entropy")}

# from sklearn.naive_bayes import GaussianNB
# clf = GaussianNB()

from sklearn import tree
dtc = tree.DecisionTreeClassifier()
clf = GridSearchCV(dtc, tree_parameters, cv=2, scoring=(scoring_options[3] + ""))
# clf = dtc

# from sklearn.svm import SVC
# svc = SVC()
# clf = GridSearchCV(svc, svc_parameters, cv=2, scoring=(scoring_options[3] + ""))

# from  sklearn.ensemble import RandomForestClassifier
# rfc = RandomForestClassifier()
# clf = GridSearchCV(rfc, rfc_parameters, cv=2, scoring=(scoring_options[3] + ""))
# clf = rfc

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)


#### NL
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
features_train_scaled = min_max_scaler.fit_transform(features_train)
features_test_scaled = min_max_scaler.transform(features_test)

clf = clf.fit(features_train_scaled, labels_train)

means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r"
          % (mean, std * 2, params))

print "Best Parameters", clf.best_params_


pred = clf.predict(features_test_scaled)
correct = 0
for i in range(len(labels_test)):
	if pred[i] == labels_test[i]:
		correct += 1
acc = 1.0 * correct / len(labels_test)

print "Test Accuracy", acc
from sklearn.metrics import precision_score, recall_score
print "Precision", precision_score(labels_test, pred, average='binary')
print "Recall", recall_score(labels_test, pred, average='binary')











### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)