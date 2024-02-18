import pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template, request
# Flask is good to development, the WSGI (Web Server Gateway Interface) is the best choice to deploy python models in production

# Start the app
app = Flask(__name__)

# Function created to read the model or scaler
def load_file(location):
    with open(location, 'rb') as file:
        return pickle.load(file)
    
model = load_file('project_3/models/model_final.pkl')
scaler = load_file('project_3/models/scaler_final.pkl')

# Route to the welcome web page
@app.route('/')
def index():
    return render_template('project_3/templates/index.html')

# Route to the prediction function
@app.route('/predict', methods = 'POST')
def prediction():
    try:

        EnglishTestScore = float(request.form.get('English test score', 0))
        PsyTestScore = int(request.form.get('Psychotechnical test score', 0))
        QIValue = int(request.form.get('QI value', 0))

        # Input validation
        if not (0 <= EnglishTestScore <= 10 and 0 <= QIValue <= 200 and 0 <= PsyTestScore <= 100):
            raise ValueError("Invalid inputs")
        
        inputs = np.array([EnglishTestScore, QIValue, PsyTestScore]).reshape(1,3)
        inputs_df = pd.DataFrame(inputs, columns=['EnglishTestScore','QIValue', 'PsyTestScore'])
   
        inputs_scaled = scaler.transform(inputs_df)
        pred = model.predict(inputs_scaled)

        output = "The student can be registered!" if pred[0] == 1 else "The student can't be registered!"
    
    except Exception as e:
        output = f"Error in the prediction: {e}"

    return render_template('index.html', result =output)

if __name__ == "__main__":
    app.run()
