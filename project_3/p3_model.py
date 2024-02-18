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