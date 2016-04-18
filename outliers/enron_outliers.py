#!/usr/bin/python

import pickle
import sys
import matplotlib.pyplot
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit


### read in data dictionary, convert to numpy array
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "r") )
data_dict.pop('TOTAL', 0)
features = ["salary", "bonus"]
data = featureFormat(data_dict, features)

poiList = []

for key in data_dict:
    value = data_dict[key]
    print 'Analyzing ', key, ' whose salary is ', value['salary'], ' bonus is ', value['bonus']
    if value['salary'] == 'NaN':
        value['salary'] = 0
    if value['bonus'] == 'NaN':
        value['bonus'] = 0
    if value['salary'] > 1000000:
        poiList.append({key: {'salary': value['salary'], 'bonus' : value['bonus']}})
    elif value['bonus'] > 7800000:
        poiList.append({key: {'salary': value['salary'], 'bonus' : value['bonus']}})

print '#########'
print poiList

### your code below
for point in data:
    salary = point[0]
    bonus = point[1]
    matplotlib.pyplot.scatter( salary, bonus )

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()

