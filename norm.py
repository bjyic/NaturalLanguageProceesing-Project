
# coding: utf-8

import requests, zipfile, StringIO, gzip
import urllib
import requests, zipfile, StringIO, gzip
import gzip
import re
import numpy as np
import pandas as pd
import ujson as json
import scipy as scp
import dill, os
import toolz
import heapq
import ujson as json
import scipy as scp
import dill, os
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn import pipeline
from data import test_json
from sklearn import pipeline
from sklearn.linear_model import Ridge
from sklearn import grid_search
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDRegressor

testfile = urllib.URLopener()
testfile.retrieve("http://thedataincubator.s3.amazonaws.com/coursedata/mldata/yelp_train_academic_dataset_business.json.gz", "yelp_train_academic_dataset_business.json.gz")
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
data_json_str= ""
for xx in content:
    data_json_str += xx
    data_json_str += ","
data_json_str = data_json_str[:-1]
data_json_str = "[" + data_json_str + "]"

#data_json_str = "[" + ','.join(content) + "]"
# now, load it into pandas
df = pd.read_json(data_json_str)
df = df[df['business_id'].isin(id_list)]
df.shape
vectorizer = CountVectorizer()

# def review_to_words( raw_review ):
#     #
#     # 1. Remove non-letters
#     letters_only = re.sub("[^a-zA-Z']", " ", raw_review)
#     #
#     # 3. Convert to lower case, split into individual words
#     words = letters_only.lower()
#     #
#     # 6. Join the words back into one string separated by space,
#     # and return the result.
#     return words

# # Initialize an empty list to hold the clean reviews
# clean_train_reviews = []

# for i in range(len(df)):
#     clean_train_reviews.append(review_to_words(df.text[i]))


# In[6]:

train_data_features = vectorizer.fit_transform(df.text)


# In[9]:

from sklearn import pipeline
from sklearn.linear_model import Ridge
from sklearn import grid_search

# In[43]:

from sklearn.feature_extraction.text import TfidfVectorizer
norm_pipe = pipeline.Pipeline([
  ('vectorizer', TfidfVectorizer(stop_words='english', min_df=200, max_df = 10000)),
  ('estimator', Ridge())
  ])
print "norm_pipe done"
norm_pipe.fit(df.text,df.stars)
print "fitting done"
dill.dump(norm_pipe, open("norm_pipe", 'wb'))
