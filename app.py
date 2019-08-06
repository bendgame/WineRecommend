import os
 
import pandas as pd
import numpy as np
import sqlite3 as sql
import seaborn as sns
import matplotlib.pyplot as plt
import pickle


import sqlalchemy 
from sqlalchemy.ext.automap import automap_base
from sqlalchemy.orm import Session
from sqlalchemy import create_engine
from flask import Flask, jsonify, render_template, request, url_for
from flask_sqlalchemy import SQLAlchemy
from sklearn.model_selection import train_test_split
from sklearn import ensemble
from sklearn.metrics import mean_absolute_error
#from sklearn.externals import joblib
from sklearn.metrics import roc_curve, auc
import joblib
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

wine_df = pd.read_sql('Select * from wine_data', db.session.bind)
g=tf.Graph()
with g.as_default():
    text_input = tf.placeholder(dtype = tf.string, shape=[None])
    embed = tfhub.Module("C:/Users/bendgame/Downloads/1fb57c3ffe1a38479233ee9853ddd7a8ac8a8c47")
    em_txt = embed(text_input)
    init_op = tf.group([tf.global_variables_initializer(), tf.tables_initializer()])
#g.finalize()

session = tf.Session(graph = g)
session.run(init_op)

result = session.run(em_txt, feed_dict={text_input:list(wine_df.description)})

@app.route("/")
def index():
    """Return the homepage."""
       
    return render_template("index.html")
  
@app.route("/FeatureEngineering", methods=['GET', 'POST'])
def FeatureEngineering():
    """Return the homepage."""
       
    return render_template("FeatureEngineering.html")

@app.route("/MRPrice", methods=['GET', 'POST'])
def MRPrice():
    """Return the homepage."""
       
    return render_template("MRPrice.html")

@app.route("/MRRating", methods=['GET', 'POST'])
def MRRating():
    """Return the homepage."""
       
    return render_template("MRRating.html")

@app.route("/RandomForestPrice", methods=['GET', 'POST'])
def RandomForestPrice():
    """Return the homepage."""
       
    return render_template("RandomForestPrice.html")

@app.route("/RandomForestRating", methods=['GET', 'POST'])
def RandomForestRating():
    """Return the homepage."""
       
    return render_template("RandomForestRating.html")

@app.route("/GBRnotebook", methods=['GET', 'POST'])
def GBRnotebook():
    """Return the homepage."""
       
    return render_template("GBRnotebook.html")

@app.route("/RecommendationsNotebook", methods=['GET', 'POST'])
def RecommendationsNotebook():
    """Return the homepage."""
       
    return render_template("RecommendationsNotebook.html")

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

    wine_df = pd.read_sql('Select * from wine_data', db.session.bind)
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
    return recommendations.head(3).T   



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
            #prd = {"price":re}
             
        return render_template("prediction.html", predicted_price = re[0] )

@app.route("/recommend", methods=['POST'])
def recommendation():
    if request.method == 'POST':
            to_predict_list = request.form.to_dict()
            query = to_predict_list['wine_desc']
            color = to_predict_list['color']
            result = recommend_engine(query, color)
            re = result.to_dict()
            recommend_list = list(re.values())
    #return jsonify(recommend_list)
    return render_template("recommendation.html"
                , variety_0 = recommend_list[0]['variety']
                , title_0 = recommend_list[0]['title']
                , price_0 = recommend_list[0]['price']
                , color_0 = recommend_list[0]['color']
                , description_0 = recommend_list[0]['description']
                , recommendation_0 = recommend_list[0]['recommendation']
                , rating_0 = recommend_list[0]['rating']
                
                , variety_1 = recommend_list[1]['variety']
                , title_1 = recommend_list[1]['title']
                , price_1 = recommend_list[1]['price']
                , color_1 = recommend_list[1]['color']
                , description_1 = recommend_list[1]['description']
                , recommendation_1 = recommend_list[1]['recommendation']
                , rating_1 = recommend_list[1]['rating']
                 
                
                , variety_2 = recommend_list[2]['variety']
                , title_2 = recommend_list[2]['title']
                , price_2 = recommend_list[2]['price']
                , color_2 = recommend_list[2]['color']
                , description_2 = recommend_list[2]['description']
                , recommendation_2 = recommend_list[2]['recommendation']
                , rating_2 = recommend_list[2]['rating']
                )
if __name__ == "__main__":
    app.run(debug = False)
