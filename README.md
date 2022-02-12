# rotten-tomatoes-sentiment
The aim of this project is to implement a machine learning model based on Naive Bayes from scratch for a sentiment analysis task using the Rotten Tomatoes movie review dataset. Obstacles like sentence negation, sarcasm, terseness, language ambiguity, and many others make this task very challenging.

The training, dev and test set contain respectively 7529, 1000 and 3310 sentences. The sentences are labelled on a scale of five values:
  0. negative
  1. somewhat negative
  2. neutral
  3. somewhat positive
  4. positive

Two different models can be trained. One for the 5-value set of labels above, another will map these sentiment values onto a 3-value scale. To switch between models, simply change line 9 in nb.py:

```
NUM_SENTIMENTS = 3 # change no. of classes here
```

In utils.py, methods of feature extraction were implemented which select only adjectives and adverbs from the movie reviews. This can be turned on and off by editing line 10 of nb.py:

```
FEATURE_EXTRACTION = False # turn feature extraction ON/OFF
```

### Results
3-value scale - Accuracy: 64.5%   Macro-F1: 0.52
5-value scale - Accuracy: 40.4%   Macro-F1: 0.34
Confusion matrices showing the performence of each model can be found in the figures folder.
