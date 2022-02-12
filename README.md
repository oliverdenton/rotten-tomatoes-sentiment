# rotten-tomatoes-sentiment
The aim of this project is to implement a machine learning model based on Naive Bayes for a
sentiment analysis task using the Rotten Tomatoes movie review dataset. Obstacles like sen-
tence negation, sarcasm, terseness, language ambiguity, and many others make this task very
challenging.

The training, dev and test set contain respectively 7529, 1000 and 3310 sentences. The sentences
are labelled on a scale of five values:

  0. negative
  1. somewhat negative
  2. neutral
  3. somewhat positive
  4. positive

Two different models can be trained. One for the 5-value set of labels above, another will map these sentiment values onto a 3-value scale. To switch between models, simply change line 9:

```
NUM_SENTIMENTS = 3 # change no. of classes here
```
