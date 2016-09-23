# coding: utf-8
import requests, zipfile, StringIO, gzip
import urllib
import gzip
import re
import numpy as np
import pandas as pd
from pandas import DataFrame
import ujson as json
import scipy as scp
import dill, os
import toolz
import heapq
from data import test_json
from sklearn import pipeline
from sklearn.linear_model import Ridge
from sklearn import grid_search
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn import pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDRegressor
testfile = urllib.URLopener()
testfile.retrieve("http://thedataincubator.s3.amazonaws.com/coursedata/mldata/yelp_train_academic_dataset_business.json.gz","yelp_train_academic_dataset_business.json.gz")
with gzip.open('yelp_train_academic_dataset_review.json.gz', 'rb') as f:
    content = f.readlines()
with gzip.open('yelp_train_academic_dataset_business.json.gz', 'rb') as f:
    lines=f.readlines()
id_list=[]
for item in lines:
    temp=json.loads(item)
    if re.search(r"restaurant", ''.join(temp["categories"]).lower()):
        id_list.append(temp["business_id"].strip())

content = map(lambda x: x.rstrip(), content)
data_json_str = ""
for xx in content:
    data_json_str += xx
    data_json_str +=","
    # ss = xx+","
data_json_str = data_json_str[:-1]
data_json_str = "["+data_json_str+"]"
#df = DataFrame.from_csv(data_json_str, sep=",")
df = pd.read_json(data_json_str)
#print type(df)
#print df.head()
df = df[df['business_id'].isin(id_list)]
df.shape
vectorizer = CountVectorizer()
train_data_features = vectorizer.fit_transform(df.text)
bag_pipe = pipeline.Pipeline([('vectorizer', CountVectorizer(stop_words='english', min_df=200, max_df = 10000)),('estimator', Ridge())])
gs = grid_search.GridSearchCV(bag_pipe,{'estimator__alpha': [0.0001,0.001,0.01]}, cv=5, n_jobs=3, scoring='mean_squared_error')
gs.fit(df.text, df.stars)
bag_pipe.fit(df.text,df.stars)
dill.dump(bag_pipe, open("bag_pipe", 'wb'))
