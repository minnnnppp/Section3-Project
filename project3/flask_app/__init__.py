from crypt import methods
from distutils.log import debug
import imp
from pickle import TRUE
from flask import Flask, render_template, request
import numpy as np
import pickle
import pandas as pd

model = None
with open("model_cat.pkl", "rb") as pk_file:
    model = pickle.load(pk_file)

app = Flask(__name__)

@app.route('/')
def index():
  return render_template('main.html')

@app.route('/predict')
def home():
    return render_template('predict.html')

@app.route('/result', methods=["POST"])
def res():
    if request.method=="POST":
        data1 = int(request.form['gender'])
        data2 = int(request.form['age'])
        data3 = int(request.form['smoke'])
        data4 = int(request.form['yellowf'])
        data5 = int(request.form['anxiety'])
        data6 = int(request.form['peerp'])
        data7 = int(request.form['chronic'])
        data8 = int(request.form['fatigue'])
        data9 = int(request.form['allergy'])
        data10 = int(request.form['wheezing'])
        data11 = int(request.form['alcohol'])
        data12 = int(request.form['cough'])
        data13 = int(request.form['breathe'])
        data14 = int(request.form['swallow-diff'])
        data15 = int(request.form['chestpain'])
        arr = np.array([[data1, data2, data3, data4, data5, data6, data7, data8, data9, data10, data11, data12, data13, data14, data15]])
        columns_name = ['GENDER', 'AGE', 'SMOKING', 'YELLOW_FINGERS','ANXIETY', 'PEER_PRESSURE', 'CHRONICDISEASE', 'FATIGUE', 'ALLERGY', 'WHEEZING', 'ALCOHOLCONSUMING', 'COUGHING', 'SHORTNESSOFBREATH', 'SWALLOWINGDIFFICULTY', 'CHESTPAIN']
        dataall = pd.DataFrame(arr, columns=columns_name)
        pred = model.predict(dataall)
        return render_template('result.html', pred=pred)
    else:
        return render_template('result.html')

if __name__ == '__main__':
    app.run(debug=True)