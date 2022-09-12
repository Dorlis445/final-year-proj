
from flask import Flask,request,render_template

import pickle
import joblib
import numpy as np
import pandas as pd


app = Flask(__name__)

model = joblib.load("model.pkl")


@app.route('/')
def home():
    return render_template("index.html")


@app.route('/predict',methods = ['POST'])
def predict():
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]

    output = model.predict(final_features)

    if output == 1:
        predict_text = "Employee likely to leave "

    else:

        predict_text = "Employee not leaving "


 
    return render_template('index.html',predict_text="Prediction: {}".format(predict_text))

if __name__ == '__main__':
    app.run(debug=True)



