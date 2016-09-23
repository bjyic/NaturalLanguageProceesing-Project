# NaturalLanguageProceesing-Project
Unstructured data makes up the vast majority of data.  This is a basic intro to
handling unstructured data.  Our objective is to be able to extract the
sentiment (positive or negative) from review text.  We will do this from Yelp
review data.

Your model will be assessed based on how root mean squared error of the number
of stars you predict.  There is a reference solution (which should not be too
hard to beat).  The reference solution has a score of 1.





# Submission                                                                                                                                                                                                 
Replace the return values in `__init__.py` with predictions from your models.
Avoid running "on-the-fly" computations or scripts in this file. Ideally you
should load your pickled model from file (in the global scope) then call
`model.predict(record)`.

# Questions

Note that all functions take an argument `record`. Samples of `record` are
given in `test_json.py`. Your model will be passed a single record during
grading.

## bag_of_words_model
Build a linear model based on the count of the words in each document
(bag-of-words model).


 

## normalized_model
Normalization is key for good linear regression. Previously, we used the count
as the normalization scheme.  Try some of these alternative vectorizations:

1. You can use the "does this word present in this document" as a normalization
   scheme, which means the values are always 1 or 0.  So we give no additional
   weight to the presence of the word multiple times.

2. Try using the log of the number of counts (or more precisely, $log(x+1)$).
   This is often used because we want the repeated presence of a word to count
   for more but not have that effect tapper off.

3. [TFIDF](https://en.wikipedia.org/wiki/Tf%E2%80%93idf) is a common
   normalization scheme used in text processing.  Use the `TFIDFTransformer`.
   There are options for using `idf` and taking the logarithm of `tf`.  Do
   these significantly affect the result?

Finally, if you can't decide which one is better, don't forget that you can
combine models with a linear regression.


## bigram_model
In a bigram model, let's consider both single words and pairs of consecutive
words that appear.  This is going to be a much higher dimensional problem
(large $p$) so you should be careful about overfitting.

Sometimes, reducing the dimension can be useful.  Because we are dealing with a
sparse matrix, we have to use `TruncatedSVD`.  If we reduce the dimensions, we
can use a more sophisticated models than linear ones.

As before, memory problems can crop up due to the engineering constraints.
Playing with the number of features, using the `HashingVectorizer`,
incorporating `min_df` and `max_df` limits, and handling stop-words in some way
are all methods of addressing this issue. If you are using `CountVectorizer`,
it is possible to run it with a fixed vocabulary (based on a training run, for
instance). Check the documentation.

*** A side note on multi-stage model evaluation: When your model consists of a
pipeline with several stages, it can be worthwhile to evaluate which parts of
the pipeline have the greatest impact on the overall accuracy (or other metric)
of the model. This allows you to focus your efforts on improving the important
algorithms, and leaving the rest "good enough".

One way to accomplish this is through ceiling analysis, which can be useful
when you have a training set with ground truth values at each stage. Let's say
you're training a model to extract image captions from websites and return a
list of names that were in the caption. Your overall accuracy at some point
reaches 70%. You can try manually giving the model what you know are the
correct image captions from the training set, and see how the accuracy improves
(maybe up to 75%). Alternatively, giving the model the perfect name parsing for
each caption increases accuracy to 90%. This indicates that the name parsing is
a much more promising target for further work, and the caption extraction is a
relatively smaller factor in the overall performance.

If you don't know the right answers at different stages of the pipeline, you
can still evaluate how important different parts of the model are to its
performance by changing or removing certain steps while keeping everything
else constant. You might try this kind of analysis to determine how important
adding stopwords and stemming to your NLP model actually is, and how that
importance changes with parameters like the number of features.

## food_bigrams
Look over all reviews of restaurants (you may need to look at the dataset from
`ml.py` to figure out which ones correspond to restaurants). We want to find
collocations --- that is, bigrams that are "special" and appear more often than
you'd expect from chance.  We can think of the corpus as defining an empirical
distribution over all ngrams.  We can find word pairs that are unlikely to
occur consecutively based on the underlying probability of their words.
Mathematically, if $p(w)$ be the probability of a word $w$ and $p(w_1 w_2)$ is
the probability of the bigram $w_1 w_2$, then we want to look at word pairs
$w_1 w_2$ where the statistic

  $$ p(w_1 w_2) / (p(w_1) * p(w_2)) $$

is high.  Return the top 100 (mostly food) bigrams with this statistic with
the 'right' prior factor (see below).


*Implementation notes:*
- The reference solution is not an aggressive filterer. Although there are
  definitely artifacts in the bigrams you'll find, many of the seeming nonsense
  words are actually somewhat meaningful and so using smoothing parameters in
  the thousands or a high min_df might give you different results.
