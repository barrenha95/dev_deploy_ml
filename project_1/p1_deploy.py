# Project 1 - Construction and Deployment of Machine Learning Models
## Objective of this script: Creation of the software that will deploy the model.

from flask import Flask, render_template, request, jsonify
import joblib

app = Flask(__name__)

# Load of the model and the transformations
dsa_model = joblib.load('project1/models/logistic_model.pkl')
transformation_package_type = joblib.load('project1/models/transformation_package_type.pkl')
transformation_product_type = joblib.load('project1/models/transformation_product_type.pkl')

# Assign the principal route for the homepage and allow only "GET" requisitions.
@app.route('/', methods = ['GET'])
def index():

    # Render the homepage using the template.html
    return render_template('template.html')

# Assign a route to make predictions and accepts only "POST" requisitions.
@app.route('/predict', methods = ['POST'])
def predict():

    # Extract the "weight" value from the sent requisition
    peso = int(request.form['Weight'])

    
