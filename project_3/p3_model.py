import sklearn
import pickle # used to save the model in disk
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

## Loading data ##
dataset = pd.read_csv('project_3\dataset.csv')
print(dataset.head())
print(dataset.describe())
print(dataset.info())
print(dataset.shape)

## Pre-processing ##

X = dataset.drop(columns = 'Admission')
y = dataset['Admission']

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=2)

# Creation of the scaler
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(X_train_scaled)
print(X_test_scaled)

print(X_train_scaled.shape)
print(X_test_scaled.shape)

## First version of the model ##
# For the first version we will use LogisticRegression because of it simplicity to explain the weights of each feature.
# It's usually the "benchmark" model because it is the more simple machine learning model.
# This model calculates the chance of the input be in each category giving a probability between 0 and 1.
# The most recommended performance metrics for this model is: Accuracy, Confusion Matrix, and AUC-ROC.

model_v1 = LogisticRegression()

model_v1.fit(X_train_scaled, y_train)
y_pred_v1 = model_v1.predict(X_test_scaled)
print(y_pred_v1)

y_pred_prob_v1 = model_v1.predict_proba(X_test_scaled)[:,1]
print(y_pred_prob_v1)

acc_v1 = accuracy_score(y_test, y_pred_v1)
print(f"\nAccuracy: {acc_v1}")

auc_v1 = roc_auc_score(y_test, y_pred_prob_v1)
print(f"\nAuc: {auc_v1}")

class_v1 = classification_report(y_test, y_pred_v1)
print(f"\Classification report: {class_v1}")

## Second version of the model ##
# For the second version we will use the RandomForest algorithm.
# The big problem of this kind of model is that he easily become overfitted.

model_v2 = RandomForestClassifier()

model_v2.fit(X_train_scaled, y_train)
y_pred_v2 = model_v2.predict(X_test_scaled)
print(y_pred_v2)

y_pred_prob_v2 = model_v2.predict_proba(X_test_scaled)[:,1]
print(y_pred_prob_v2)

acc_v2 = accuracy_score(y_test, y_pred_v2)
print(f"\nAccuracy: {acc_v2}")

auc_v2 = roc_auc_score(y_test, y_pred_prob_v2)
print(f"\nAuc: {auc_v2}")

class_v2 = classification_report(y_test, y_pred_v2)
print(f"\Classification report: {class_v2}")

## Third version of the model ##
# For the third version we will use the Support Vector Machines (SVM) algorithm
# It's a good model but it's heavier than the other and must a special attention to the hyperparameters.
# It basically draw a plan in the space that explains the best some behavior.

model_v3 = SVC(verbose=True, probability=True)
model_v3.fit(X_train_scaled, y_train)
y_pred_v3 = model_v3.predict(X_test_scaled)
print(y_pred_v3)

y_pred_prob_v3 = model_v3.predict_proba(X_test_scaled)[:, 1]
print(y_pred_prob_v3)

acc_v3 = accuracy_score(y_test, y_pred_v3)
print(f"\nAccuracy: {acc_v3}")

auc_v3 = roc_auc_score(y_test, y_pred_prob_v3)
print(f"\nAuc: {auc_v3}")

class_v3 = classification_report(y_test, y_pred_v3)
print(f"\Classification report: {class_v3}")