#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))


print len(enron_data)

keys = enron_data.keys()

print keys
print len(enron_data[keys[0]]), len(enron_data[keys[1]])
print enron_data[keys[0]]

pois = 0
for k in keys:
	if enron_data[k]["poi"]:
		pois += 1
print pois

print enron_data["SKILLING JEFFREY K"]

max_pay = 0
who_did = ""
for k in keys:
	if k != "TOTAL" and enron_data[k]["total_payments"] != "NaN" and enron_data[k]["total_payments"] > max_pay:
		max_pay = enron_data[k]["total_payments"]
		who_did = k
print max_pay, who_did

hav_qsal = 0
hav_email = 0
nan_total_pay = 0
nan_total_pay_poi = 0
for k in keys:
	if enron_data[k]["salary"] != "NaN":
		hav_qsal += 1
	if enron_data[k]["email_address"] != "NaN":
		hav_email += 1
	if enron_data[k]["total_payments"] == "NaN":
		nan_total_pay += 1
		if enron_data[k]["poi"]:
			nan_total_pay_poi += 1
print hav_qsal, hav_email
print nan_total_pay, 1.0 * nan_total_pay / len(enron_data)
print nan_total_pay_poi