import nltk
import numpy as np
import pandas as pd
import utils
from collections import defaultdict
import matplotlib.pyplot as plt
import itertools

NUM_SENTIMENTS = 3 # change no. of classes here
FEATURE_EXTRACTION = False # turn feature extraction ON/OFF
SENTIMENTS_MAP = {0: 0, 1: 0, 2: 1, 3: 2, 4: 2}


def get_frequencies(train_df):
    """ Builds frequency data structures for training
    param: train_df: the training dataset
    return: occurences of each sentiment value,
            word frquencies dict for each sentiment value
    """
    s_counts = [] # Index is sentiment value
    freq_dict = {} # (word, sentiment) -> frequency

    for sentiment in range(NUM_SENTIMENTS):
        df = train_df.loc[train_df["Sentiment"] == sentiment]
        s_counts.append(len(df))

        for review in df["Phrase"]:
            words = utils.preprocess_review(review, extract=FEATURE_EXTRACTION)

            for w in words:
                pair = (w, sentiment)
                if pair in freq_dict:
                    freq_dict[pair] += 1
                else:
                    freq_dict[pair] = 1
    return s_counts, freq_dict


def train(s_counts, freqs):
    """ Trains a naive bayes model
    param: s_counts: occurences for each sentiment value
    param: freqs: word frquencies dict for each sentiment value
    return: prior probabilities of each class,
            likelihoods of each word appearing in each class
    """
    total = sum(s_counts)
    priors = []
    for count in s_counts:
        priors.append(count / total)

    # Gets the total no. of w ords for each sentiment value
    num_sentiment_words = []
    for i in range(NUM_SENTIMENTS):
        num_sentiment_words.append(0)

    for pair in freqs.keys():
        num_sentiment_words[pair[1]] += freqs[(pair)]

    # Gets the no. of unique words in the dataset
    unique = set([pair[0] for pair in freqs.keys()])
    num_unique = len(unique)

    likelihoods = defaultdict(dict) # {word -> {sentiment -> probability}}
    for word in unique:
        for sentiment in range(NUM_SENTIMENTS):
            # Gets no. of times a given word appears in review with sentiment S
            w_given_s_freq = freqs.get((word, sentiment), 0) 
            # Laplace smoothing
            p_w_given_s = (w_given_s_freq + 1) / (num_sentiment_words[sentiment] + num_unique)
            likelihoods[word][sentiment] = p_w_given_s

    return priors, likelihoods


def predict(review, priors, likelihoods):
    """ Predicts the sentiment of a given review
    param: review: the review to be analysed
    param: priors: prior probabilities of each class
    param: likelihoods: likelihoods of each word appearing in each class
    return: predicted class (sentiment value)
    """
    review = utils.preprocess_review(review, extract=FEATURE_EXTRACTION)
    trained_words = likelihoods.keys()
    probabilities = []

    for sentiment in range(NUM_SENTIMENTS):
        product = priors[sentiment]
        for word in review:
            if word in trained_words:
                product *= likelihoods[word][sentiment]
        probabilities.append(product)

    prediction = probabilities.index(max(probabilities))
    return prediction


def plot_confusion(cm):
    """ Creates confusion matrix plot and displays it
    param: cm: the confusion matrix to plot
    """
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    cmap = plt.get_cmap('PuRd')
    # Changes plot depending on no. of classes
    if NUM_SENTIMENTS == 3:
        plt.figure(figsize=((8/5)*3, (6/5)*3)) 
        target_names = ["Negative", "Neutral", "Positive"]
    else:
        plt.figure(figsize=(8,6))
        target_names = ["Negative", "Somewhat Negative", "Neutral", "Somewhat Positive", "Positive"]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title("Confusion Matrix")
    plt.colorbar()

    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names, rotation=45)
    plt.yticks(tick_marks, target_names)

    thresh = cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, "{:,}".format(cm[i, j]),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")

    plt.grid(False)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted Label\nAccuracy={:0.4f}; Misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()


def calc_f1(cm):
    """ Calculates macro and micro F1 scores
    param: cm: confusion matrix
    """
    f1_scores = []
    for i in range(NUM_SENTIMENTS):
        TP = cm[i][i]
        FP = 0
        FN = 0
        for j in range(NUM_SENTIMENTS):
            if i != j:
                FP += cm[j][i]
                FN += cm[i][j]
        f1_scores.append((2*TP) / (2* TP + FP + FN))

    return (sum(f1_scores) / NUM_SENTIMENTS), f1_scores


def test():
    train_df = pd.read_table("moviereviews/train.tsv")
    test_df = pd.read_table("moviereviews/dev.tsv")

    # Maps 5-class problem to 3-class problem
    if NUM_SENTIMENTS == 3:
        train_df["Sentiment"].replace(SENTIMENTS_MAP, inplace=True)
        test_df["Sentiment"].replace(SENTIMENTS_MAP, inplace=True)
    
    # Training functions
    s_counts, freqs = get_frequencies(train_df)
    priors, likelihoods = train(s_counts, freqs)

    # Initialise results variables
    res = []
    total = len(test_df)
    correct = 0
    cm = np.zeros((NUM_SENTIMENTS, NUM_SENTIMENTS))

    # Obtaining results
    for review in test_df["Phrase"]:
        prediction = predict(review, priors, likelihoods)
        idx = test_df.loc[test_df["Phrase"] == review, "SentenceId"].iloc[0]   
        res.append([idx, prediction])

        real = test_df.loc[test_df["Phrase"] == review, "Sentiment"].iloc[0]        
        cm[real][prediction] += 1
        if real == prediction:
            correct += 1

    # Output results
    # res_df = pd.DataFrame(res, columns=["SentenceId", "Sentiment"])
    # res_df.to_csv(f"/Users/oliverdenton/Text/assign2/predictions/test_predictions_{NUM_SENTIMENTS}classes_Oliver_DENTON.tsv",
    #                 sep="\t", index=False)

    print(f"Accuracy: {correct / total}")
    f1, f1_scores = calc_f1(cm)
    print(f"F1: {f1}")
    print(f"Micro F1s: {f1_scores}") # Index is sentiment value
    plot_confusion(cm)

test()