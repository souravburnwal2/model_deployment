from flask import Flask,request,jsonify, render_template

import pickle
import numpy as np

model_path = 'premium.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)

app= Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract data from form
    input_data = [int(x) for x in request.form.values()]
    age=np.array(input_data).reshape(1,-1)

    # make prediction
    prediction = model.predict(age)[0]

    return render_template('index.html', premium = 'Predicted Premium: ${}'.format(prediction))

if __name__== '__main__':
    app.run(debug=True)