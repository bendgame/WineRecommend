#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import re
import tensorflow as tf
import tensorflow_hub as tfhub
import sqlite3
from sqlite3 import Error

pd.set_option('display.max_colwidth', -1)


# In[2]:


conn = sqlite3.connect('wine_data.sqlite')
c = conn.cursor()


# In[22]:


wine_df = pd.read_sql('Select * from wine_data', conn)


# In[42]:


wine_df.shape
wine_df.head(1)


# In[5]:


embed = tfhub.Module("C:/Users/Administrator/Downloads/1fb57c3ffe1a38479233ee9853ddd7a8ac8a8c47")


# In[9]:


#download the model to local so it can be used again and again
#!mkdir ../wine_model/module_useT
# Download the module, and uncompress it to the destination folder. 
#!curl -L "https://tfhub.dev/google/universal-sentence-encoder-large/3?tf-hub-format=compressed" | tar -zxvC ../wine_model/module_useT


# In[6]:



wd = list(wine_df.description)
output = embed(wd)


# In[7]:


#embed = tfhub.Module("C:/Users/Administrator/Downloads/1fb57c3ffe1a38479233ee9853ddd7a8ac8a8c47")

with tf.Session() as session:
    session.run([tf.global_variables_initializer(), tf.tables_initializer()])
    wine_embeddings = session.run(output)
    print(wine_embeddings.shape)


# In[18]:



g=tf.Graph()
with g.as_default():
text_input = tf.placeholder(dtype = tf.string, shape=[None])
embed = tfhub.Module("C:/Users/Administrator/Downloads/1fb57c3ffe1a38479233ee9853ddd7a8ac8a8c47")
em_txt = embed(text_input)
init_op = tf.group([tf.global_variables_initializer(), tf.tables_initializer()])
#g.finalize()

session = tf.Session(graph = g)
session.run(init_op)


# In[25]:


result = session.run(em_txt, feed_dict={text_input:list(wine_df.description)})


# In[50]:


def recommend_engine(query, color, embedding_table = result):
    # Embed user query
    with tf.Session(graph = g) as session:
        session.run([tf.global_variables_initializer(), tf.tables_initializer()])
        embedding = session.run(embed([query]))

    # Calculate similarity with all reviews
    dot = np.dot(embedding, embedding_table.T)
    norm_a = np.linalg.norm(embedding)
    norm_b = np.linalg.norm(embedding_table.T)
    similarity_score = dot/(norm_a * norm_b)
   
    recommendations = wine_df.copy()
    recommendations['recommendation'] = similarity_score.T
    recommendations['dot'] = dot.T
    recommendations = recommendations.sort_values('recommendation', ascending=False)
    
    
    if (color == 'red'):
        recommendations = recommendations.loc[(recommendations.color =='red')] 
        recommendations = recommendations[['variety', 'title', 'price', 'description', 'recommendation'
                                       , 'rating','dot','color']]
    elif(color == "white"):
        recommendations = recommendations.loc[(recommendations.color =='white')] 
        recommendations = recommendations[['variety', 'title', 'price', 'description', 'recommendation'
                                       , 'rating','dot','color']]
    elif(color == "other"):
        recommendations = recommendations.loc[(recommendations.color =='other')] 
        recommendations = recommendations[['variety', 'title', 'price', 'description', 'recommendation'
                                       , 'rating','dot','color']]
    else:
        recommendations = recommendations[['variety', 'title', 'price', 'description', 'recommendation'
                                       , 'rating','dot','color']]

    return recommendations


# In[32]:


import time


# In[51]:



query = "Low Tanin, violet, passion fruit"
color = 'all'
t1 = time.time()
recommendation = recommend_engine(query, color)

print(query)

recommendation.head().T
#print(time.time() - t1)


# In[52]:


query = "Low Tanin, violet, passion fruit"
color = 'red'
t1 = time.time()
recommendation = recommend_engine(query, color)

print(query)

recommendation.head().T


# In[ ]:




