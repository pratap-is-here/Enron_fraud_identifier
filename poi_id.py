#!/usr/bin/python

import sys
import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn.cross_validation import StratifiedShuffleSplit
### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 
                 'deferred_income', 'total_stock_value', 'expenses','exercised_stock_options', 'other', 'long_term_incentive',
                 'restricted_stock', 'director_fees','to_messages', 'from_poi_to_this_person', 'from_messages',
                  'from_this_person_to_poi', 'shared_receipt_with_poi'] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)


### DATA EXPLORATION
### number of data points
count_data_points=0
count_poi=0
count_features=0
x=1
for point in data_dict:
	count_data_points +=1
	count_features=len(data_dict[point])
	if data_dict[point]['poi']==True:
		count_poi+=1
        
print ("number of data point : ",count_data_points)
print ("number of poi : ",count_poi)
print ("total number of features : ",count_features)

### Task 2: Remove outliers

### read in data dictionary, convert to numpy array


features = ["salary", "bonus"]

data = featureFormat(data_dict, features)


### your code below
for point in data:
    salary = point[0]
    bonus = point[1]
    plt.scatter( salary, bonus )

plt.xlabel("salary")
plt.ylabel("bonus")
plt.show()

data_dict.pop( 'TOTAL', 0 )#this point was an outlier which was removed after careful analysis through various plots

   

### Task 3: Feature(s) Selection

#first, let us see the accuracy of a simple decision tree classifier when we use all the features we have

data=featureFormat(data_dict,features_list, sort_keys = True)
y, X = targetFeatureSplit(data)
y=np.array(y)
X=np.array(X)


#feature scalling
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
X=scaler.fit_transform(X)


X_indices = np.arange(X.shape[-1])
print "X_indices :",X_indices
# Univariate feature selection with F-test for feature scoring
# We use the default selection function: the 10% most significant features
selector = SelectPercentile(f_classif, percentile=20)
np.seterr(divide='ignore', invalid='ignore')
X_new=selector.fit_transform(X,y)
print 'shape of X_new ',np.shape(X_new)

#scores = -np.log10(selector.pvalues_)
#scores /= scores.max()
print selector.scores_
print "\n\n"
t=features_list
del t[0]
features_list=['poi']
print 'selected features:\n'
arr= np.array(selector.get_support(indices=True))
for i in arr:
	print t[i]
	features_list.append(t[i])
	print '\n'
print 'new feature list : ',np.shape(features_list)
plt.bar(X_indices , selector.scores_, width=.2,
        label=r'Univariate score ($-Log(p_{value})$)', color='darkorange',
        edgecolor='black')

plt.show()

### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
#data = featureFormat(my_dataset, features_list, sort_keys = True)
#labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html


### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html


### BOTH task 4 and 5 done together.


from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(X_new, y, test_size=0.30, random_state=42)

from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

#Trial 1: Decision Tree Classifier 

from sklearn.tree import DecisionTreeClassifier

parameters={'criterion':('gini','entropy'),'splitter':('best','random')}#OUTPUT: best parameters = ('criterion:-gini','splitter:-random')
dtc = DecisionTreeClassifier()
clf=GridSearchCV(dtc,parameters)

clf.fit(features_train, labels_train)

print("Best estimator found by grid search:")
print(clf.best_estimator_)
pred=clf.predict(features_test)

#Trial 2: SVC
from sklearn.svm import SVC

#After tuning the parameters (C,kernel and gamma), following was the best result. Accuracy wasn't much affected although precision and
#recall improved. 
clf=SVC(C=100,kernel='linear')
clf.fit(features_train, labels_train)
pred=clf.predict(features_test)
print "\naccuracy_score : ",accuracy_score(labels_test,pred)#OUTPUT: 88.6 %
print '\nno. of predicted POIs : ',sum(pred)#OUTPUT: 2
print '\nno. of people in the dataset : ',len(pred)
#print accuracy_score(labels_test,[0]*29)
for i,j in zip(pred,labels_test):
    print 'predicted : ',i,' but actual : ',j
    if i==1 and j==1:
        print 'this one is true positive :)'

#precision and recall nowwww

print '\nprecision : \n',precision_score(labels_test,pred)#OUTPUT: 0.5
print '\nrecall : \n',recall_score(labels_test,pred)#OUTPUT: 0.2

PERF_FORMAT_STRING = "\
\tAccuracy: {:>0.{display_precision}f}\tPrecision: {:>0.{display_precision}f}\t\
Recall: {:>0.{display_precision}f}\tF1: {:>0.{display_precision}f}\tF2: {:>0.{display_precision}f}"
RESULTS_FORMAT_STRING = "\tTotal predictions: {:4d}\tTrue positives: {:4d}\tFalse positives: {:4d}\
\tFalse negatives: {:4d}\tTrue negatives: {:4d}"
folds=1000
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
cv = StratifiedShuffleSplit(labels, folds, random_state = 42)
true_negatives = 0
false_negatives = 0
true_positives = 0
false_positives = 0
for train_idx, test_idx in cv: 
    features_train = []
    features_test  = []
    labels_train   = []
    labels_test    = []
    for ii in train_idx:
        features_train.append( features[ii] )
        labels_train.append( labels[ii] )
    for jj in test_idx:
        features_test.append( features[jj] )
        labels_test.append( labels[jj] )
        
    ### fit the classifier using training set, and test on test set
    clf.fit(features_train, labels_train)
    predictions = clf.predict(features_test)
    for prediction, truth in zip(predictions, labels_test):
        if prediction == 0 and truth == 0:
            true_negatives += 1
        elif prediction == 0 and truth == 1:
            false_negatives += 1
        elif prediction == 1 and truth == 0:
            false_positives += 1
        elif prediction == 1 and truth == 1:
            true_positives += 1
        else:
            print("Warning: Found a predicted label not == 0 or 1.")
            print("All predictions should take value 0 or 1.")
            print("Evaluating performance for processed predictions:")
            break
try:
    total_predictions = true_negatives + false_negatives + false_positives + true_positives
    accuracy = 1.0*(true_positives + true_negatives)/total_predictions
    precision = 1.0*true_positives/(true_positives+false_positives)
    recall = 1.0*true_positives/(true_positives+false_negatives)
    f1 = 2.0 * true_positives/(2*true_positives + false_positives+false_negatives)
    f2 = (1+2.0*2.0) * precision*recall/(4*precision + recall)
    print(clf)
    print(PERF_FORMAT_STRING.format(accuracy, precision, recall, f1, f2, display_precision = 5))
    print(RESULTS_FORMAT_STRING.format(total_predictions, true_positives, false_positives, false_negatives, true_negatives))
    print("")
except:
    print("Got a divide by zero when trying out:", clf)
    print("Precision or recall may be undefined due to a lack of true positive predicitons.")


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
