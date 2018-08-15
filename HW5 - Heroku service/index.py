import os
import numpy as np
import pandas as pd
from sklearn.externals import joblib
from flask import Flask, request
from transformers import MyRegression, MyXGBoost, GeoDataTransformer
import locale
locale.setlocale(locale.LC_ALL, 'en_US.utf8')

app = Flask(__name__)
model = joblib.load('model.pkl')

@app.route('/')
def welcome():
    return '''
        <html>
        <head>
            <style>
                * {
                    font-family: Sans
                }
                
                img {
                    display: block;
                    margin: 0 auto;
                    max-width: 600
                }
                
                h1, h3 {
                    text-align: center;
                }
            </style>
        </head>
        <body>
            <img src="https://www.thenational.ae/image/policy:1.186093:1499305668/image/jpeg.jpg?f=16x9&w=1200&$p$f$w=dfa40e8" />
            <h1>Russian Housing Market</h1>
            <h3>
                Example query:
                <a href="/predict?full_sq=65&life_sq=39&build_year=1990&num_room=3&sub_area=Zamoskvorech">
                    /predict?full_sq=65&life_sq=39&build_year=1990&num_room=3&sub_area=Zamoskvorech
                </a>
            </h3>
        </body>
        '''


@app.route('/predict')
def predict():
    full_sq = request.args.get('full_sq', type=float) or 'NaN'
    life_sq = request.args.get('life_sq', type=float) or 'NaN'
    build_year = request.args.get('build_year', type=float) or 'NaN'
    num_room = request.args.get('num_room', type=float) or 'NaN'
    sub_area = request.args.get('sub_area', type=str) or 'NaN'
    columns = ['full_sq', 'life_sq', 'build_year', 'num_room', 'sub_area']
    prediction = model.predict(pd.DataFrame([[full_sq, life_sq, build_year, num_room, sub_area]], columns=columns))
    prediction = locale.format("%d", prediction, grouping=True)
    return '''
        <html>
        <head>
            <style>
                * {
                    font-family: Sans;
                }
                
                img {
                    display: block;
                    margin: 0 auto;
                    max-width: 600;
                }
                
                h1, h3 {
                    text-align: center;
                }
                
                span {
                    color: #00C5CD
                }
            </style>
        </head>
         <body>
            <img src="https://cdn.houseplans.com/product/q5qkhirat4bcjrr4rpg9fk3q94/w1024.jpg?v=8" />
            <h1>Russian Housing Market</h1>
            <h3>Predicted price: <span>%s&#8381</span></h3>
        </body>
        <html>
        ''' % prediction

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
