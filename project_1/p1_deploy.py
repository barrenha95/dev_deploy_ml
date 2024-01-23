# Project 1 - Construction and Deployment of Machine Learning Models
## Objective of this script: Creation of the software that will deploy the model.

from flask import Flask, render_template, request, jsonify
import joblib

app = Flask(__name__)

# Load of the model and the transformations
dsa_model = joblib.load('project1/models/logistic_model.pkl')
le_package_type = joblib.load('project1/models/transformation_package_type.pkl')
le_product_type = joblib.load('project1/models/transformation_product_type.pkl')

# Assign the principal route for the homepage and allow only "GET" requisitions.
@app.route('/', methods = ['GET'])
def index():

    # Render the homepage using the template.html
    return render_template('template.html')

# Assign a route to make predictions and accepts only "POST" requisitions.
@app.route('/predict', methods = ['POST'])
def predict():

    # Extract the "weight" value from the sent requisition
    weight = int(request.form['Weight'])

    # Tranform the package_type using the label encoder previously fited
    package_type = le_package_type.transform([request.form['package_type']])[0]

    # Use the model to make the prediction
    prediction = dsa_model.predict([[weight, package_type]])[0]

    # Convert the encoded prediction to it original label
    product_type = le_product_type.inverse_transform([prediction])[0]

    # Render the homepage with the prediction
    return render_template('template.html', prediction = product_type)

# App
## This is the key that says it is a software instead of a script
### Or in other words: It MUST be runned as only one thing
if __name__ == '__main__':
    app.run()
