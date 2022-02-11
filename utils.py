import nltk
from nltk.tokenize import RegexpTokenizer
import numpy as np
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from nltk import pos_tag

tokenizer = RegexpTokenizer(r'\w+')
nltk.download("stopwords")
stopwords = stopwords.words("english")
stemmer = PorterStemmer()
nltk.download('averaged_perceptron_tagger')


# Splits sentence into words or 'tokens'
def tokenize(review):
    return tokenizer.tokenize(review)


# Removes words present in the stop list
def apply_stoplist(tokens):
    review_clean = []
    for word in tokens:
        if (word not in stopwords):
            review_clean.append(word)
    return review_clean


# Retains only adjectives and adverbs
def extract_features(tokens):
    tags = pos_tag(tokens)
    features = []
    for t in tags:
        # Adjective = JJ and Adverb == RB
        if t[1] == 'JJ' or t[1] == 'RB':
            features.append(t[0])
    return features


# Reduces a given word to its stem
def stem(words):
    stemmed = []
    for w in words:
        stemmed.append(stemmer.stem(w.lower()))
    return stemmed


# Combines all pre-processing steps
def preprocess_review(review, extract):
    tokens = tokenize(review)
    clean = apply_stoplist(tokens)
    if extract:
        features = extract_features(clean)
    stemmed = stem(clean)
    return stemmed