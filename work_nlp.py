import requests, zipfile, StringIO, gzip
import urllib

testfile = urllib.URLopener()
testfile.retrieve("http://thedataincubator.s3.amazonaws.com/coursedata/mldata/yelp_train_academic_dataset_business.json.gz", "yelp_train_academic_dataset_business.json.gz")


import gzip
with gzip.open('yelp_train_academic_dataset_review.json.gz', 'rb') as f:
    content = f.readlines()

import ujson as json
import re
with gzip.open('yelp_train_academic_dataset_business.json.gz', 'rb') as f:
    lines=f.readlines()
id_list=[]
for item in lines:
    temp=json.loads(item)
    if re.search(r"restaurant", ''.join(temp["categories"]).lower()):
        id_list.append(temp["business_id"].strip())

import numpy as np
import pandas as pd
import scipy as scp
import dill, os
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn import pipeline


# strip \n from each line
content = map(lambda x: x.rstrip(), content)

data_json_str = "[" + ','.join(content) + "]"

# now, load it into pandas
df = pd.read_json(data_json_str)


# In[8]:

df = df[df['business_id'].isin(id_list)]


# In[10]:

df.shape


# In[4]:

import re
from sklearn.feature_extraction.text import CountVectorizer
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

# pipeline for bag of words
bag_pipe = pipeline.Pipeline([
  ('vectorizer', CountVectorizer(stop_words='english', min_df=200, max_df = 10000)),
  ('estimator', Ridge())
  ])
gs = grid_search.GridSearchCV(
    bag_pipe,
    {'estimator__alpha': [0.0001,0.001,0.01]},
    cv=5,  # 5 fold cross validation
    n_jobs=3,  # run each hyperparameter in one of two parallel jobs
    scoring='mean_squared_error'
)
gs.fit(df.text, df.stars)

bag_pipe.fit(df.text,df.stars)


dill.dump(bag_pipe, open("bag_pipe", 'wb'))


from sklearn.feature_extraction.text import TfidfVectorizer
norm_pipe = pipeline.Pipeline([
  ('vectorizer', TfidfVectorizer(stop_words='english', min_df=200, max_df = 10000)),
  ('estimator', Ridge())
  ])
norm_pipe.fit(df.text,df.stars)


# In[44]:

dill.dump(norm_pipe, open("norm_pipe", 'wb'))


from sklearn.linear_model import SGDRegressor
big_pipe = pipeline.Pipeline([
  ('vectorizer', TfidfVectorizer(stop_words='english',ngram_range = (1,2), min_df=200, max_df = 5000)),
  ('estimator', SGDRegressor())
  ])
big_pipe.fit(df.text,df.stars)
dill.dump(big_pipe, open("big_pipe", 'wb'))


import toolz
from data import test_json
test_json = [toolz.keyfilter(lambda k: k == "text", d)
            for d in test_json]


from sklearn.feature_extraction.text import CountVectorizer
univectorizer = CountVectorizer(stop_words='english',min_df=10)
bivectorizer = CountVectorizer(stop_words='english',ngram_range=(2,2), min_df = 10)


unimat = univectorizer.fit_transform(df.text)


bimat = bivectorizer.fit_transform(df.text)


usum = unimat.sum(axis=0)


ugrams = {}
for item in univectorizer.vocabulary_.items():
    ugrams[item[0]] = (usum[0,item[1]]+200)/float(unimat.shape[0])


unimat.shape[0]


# In[9]:

dill.dump(ugrams, open("ugrams", 'wb'))


# In[77]:

del univectorizer


# In[13]:

bisum = bimat.sum(axis=0)


# In[14]:

bigrams = []
for item in bivectorizer.vocabulary_.items():
    bigrams.append((item[0],bisum[0,item[1]]/float(bimat.shape[0])))


del bivectorizer
del bimat


import heapq
heap = []
for i,bigram in enumerate(bigrams):
    word = bigram[0].split(' ')
    try:
        p1 = ugrams[word[0]]
        p2 = ugrams[word[1]]
        den = p1*p2
        val = bigram[1]/den
    except:
        val = 0
    if i<100:
        heapq.heappush(heap,(val,bigram[0]))
    else:
        heapq.heappushpop(heap,(val,bigram[0]))

k = []
heapq.heappush(k,(5,'l'))
heapq.heappush(k,(7,'p'))
heapq.heappushpop(k,(8.5075465660557081, u'smiles males'))
n = bigrams[0][0].split(' ')
