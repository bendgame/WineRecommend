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

import tensorflow as tf
import tensorflow_hub as tfhub

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
    wine_df = pd.read_sql('Select * from wine_data', db.session.bind)
    g=tf.Graph()
    with g.as_default():
        text_input = tf.placeholder(dtype = tf.string, shape=[None])
        embed = tfhub.Module("https://tfhub.dev/google/universal-sentence-encoder/2")
        em_txt = embed(text_input)
        init_op = tf.group([tf.global_variables_initializer(), tf.tables_initializer()])
    #g.finalize()

    session = tf.Session(graph = g)
    session.run(init_op)

    result = session.run(em_txt, feed_dict={text_input:list(wine_df.description)})
    
    
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

def recommend_engine(query, color, embedding_table = result):
    '''
    takes user query, wine color, and embedded descriptions. Encodes the user query 
    and uses the dot product (calculated using numpy) to calculate the similarity 
    between the description and user query.
    '''
    
    # Embed user query
    with tf.Session(graph = g) as session:
        session.run([tf.global_variables_initializer(), tf.tables_initializer()])
        embedding = session.run(embed([query]))

    # Calculate similarity with all reviews
    similarity_score = np.dot(embedding, embedding_table.T)
     
    recommendations = wine_df.copy()
    recommendations['recommendation'] = similarity_score.T
    recommendations = recommendations.sort_values('recommendation', ascending=False)
    
    #filter through the dataframe to find the corresponding wine color records.
    if (color == 'red'):
        recommendations = recommendations.loc[(recommendations.color =='red')] 
        recommendations = recommendations[['variety', 'title', 'price', 'description', 'recommendation'
                                       , 'rating','color']]
    elif(color == "white"):
        recommendations = recommendations.loc[(recommendations.color =='white')] 
        recommendations = recommendations[['variety', 'title', 'price', 'description', 'recommendation'
                                       , 'rating','color']]
    elif(color == "other"):
        recommendations = recommendations.loc[(recommendations.color =='other')] 
        recommendations = recommendations[['variety', 'title', 'price', 'description', 'recommendation'
                                       , 'rating','color']]
    else:
        recommendations = recommendations[['variety', 'title', 'price', 'description', 'recommendation'
                                       , 'rating','color']]
    #returns dataframe
    return recommendations



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

@app.route("/recommend", methods=['POST'])
def recommendation():
    if request.method == 'POST':
            to_predict_list = request.form.to_dict()
            query = to_predict_list['wine_desc']
            color = to_predict_list['color']
            result = recommend_engine(query, color)
            

    return jsonify(result)

if __name__ == "__main__":
    app.run(debug = True)