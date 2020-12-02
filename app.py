import numpy as np
import pandas as pd
from flask import Flask, request, render_template, jsonify
import pickle  # python nesnelerini kaydetmek ve cagirmak icin kullanilir.
from flask import Flask,request, url_for, redirect, render_template, jsonify
from pycaret.regression import *
import pandas as pd
import pickle
import numpy as np


from pycaret.anomaly import predict_model

app = Flask(__name__)  # flask uygulamasını initialize eder. baslatir.
model = pickle.load(open('lrrr_model.pkl', 'rb'))  # kayitli modeli cagirma


# flask root (kök dizin) için anasayfanın tanımlanması
@app.route('/')
def home():
    return render_template('template.html')  # anasayfa icin hazirlanmis olan template belirtilir.


# yukaridaki bolum sonrasi template.html icindeki hareketleri okumaliyiz.


@app.route('/predict',methods=['POST'])
def predict():
    int_features = [x for x in request.form.values()]
    final = np.array(int_features)
    data_unseen = pd.DataFrame([final], columns = cols)
    prediction = predict_model(model, data=data_unseen, round = 0)
    prediction = int(prediction.Label[0])
    return render_template('template.html',pred='Expected Bill will be {}'.format(prediction))






@app.route('/predict_api',methods=['POST'])
def predict_api():
    data = request.get_json(force=True)
    data_unseen = pd.DataFrame([data])
    prediction = predict_model(model, data=data_unseen)
    output = prediction.Label[0]
    return jsonify(output)

if __name__ == '__main__':
    app.run(debug=True)