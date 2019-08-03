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
from flask import Flask, jsonify, render_template
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__, static_url_path='/static')



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
    return render_template("index.html")
  


@app.route("/country")
def country():
    stmnt1 = 'Select distinct country from wine_data order by country'
    df = pd.read_sql(stmnt1, db.session.bind)
    data1 = []
    a = 0
    while a < len(df):
        country = {
                'country':list(df['country'])[a],
            }
        data1.append(country)
        a+=1
    
    stmnt2 = 'Select distinct rating from wine_data'
    df2 = pd.read_sql(stmnt2, db.session.bind)
    data2 = []
    b = 0
    while b < len(df2):
        rating = {
                'rating':list(df2['rating'])[b]
                }
        data2.append(rating)
        b+=1
    return jsonify(data1)

# def rating():
#     stmnt2 = 'Select distinct rating from wine_data'
#     df2 = pd.read_sql(stmnt2, db.session.bind)
#     data2 = []
#     b = 0
#     while b < len(df2):
#         rating = {
#                 'rating':list(df2['rating'])[b]
#                 }
#         data2.append(rating)
#         b+=1
    
#     return jsonify(data2)


if __name__ == "__main__":
    app.run()


