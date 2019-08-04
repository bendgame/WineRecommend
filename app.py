import os
 
import pandas as pd
import numpy as np
import sqlite3 as sql
import seaborn as sns
import matplotlib.pyplot as plt

import sqlalchemy 
from sqlalchemy.ext.automap import automap_base
from sqlalchemy.orm import Session
from sqlalchemy import create_engine
from flask import Flask, jsonify, render_template, request
from flask_sqlalchemy import SQLAlchemy
from sklearn.model_selection import train_test_split
from sklearn import ensemble
from sklearn.metrics import mean_absolute_error
from sklearn.externals import joblib
from sklearn.metrics import roc_curve, auc

app = Flask(__name__, static_url_path='/static') 
model = joblib.load("xbgWinePrice.pkl")



#################################################
# Database Setup
#################################################

app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///db/wine_data.sqlite"
db = SQLAlchemy(app)

# reflect an existing database into a new model
Base = automap_base()
# reflect the tables
Base.prepare(db.engine, reflect=True)

@app.route("/")
def index():
    """Return the homepage."""
    return render_template("index2.html")
  


def prediction(rating, country, variety, color):

    stmnt1 = 'Select * from wine_data'
    df = pd.read_sql(stmnt1, db.session.bind)

    dfco = df.loc[(df.country == country)][:1]
    dfv = df.loc[(df.variety == variety)][:1]
    dfc = df.loc[(df.color == color)][:1]
  
    a = list(dfco['countryID'])
    b = list(dfv['varietyID'])
    c = list(dfc['colorID'])   

    dft = pd.DataFrame({"rating":rating,
                        "countryID":a,
                        "varietyID":b,
                        "colorID":c,})
    
    
    return model.predict(dft).round()

@app.route("/result", methods=['POST'])
def predict():
        if request.method == 'POST':
            to_predict_list = request.form.to_dict()
            #to_predict_list=list(to_predict_list.values())
            country = to_predict_list['country']
            color = to_predict_list['color']
            variety = to_predict_list['variety']
            rating = to_predict_list['rating']
        
            result = prediction(rating,country,variety,color)
            re = list(result)
            prd = {"price":re}
             
        return render_template("prediction.html", predicted_price = re[0] )

if __name__ == "__main__":
    app.run(debug = True)