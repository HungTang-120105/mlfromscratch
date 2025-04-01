import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
from logisticregression import LogisticRegression
import json
import re
import os

relative_path_1 = os.path.join(os.path.dirname(__file__), "amazon_baby_subset.csv")
relative_path_2 = os.path.join(os.path.dirname(__file__), "important_words.json")

def clean_sentences(string):
    label_chars = re.compile("[^A-Za-z \n]+")
    string = string.lower()
    return re.sub(label_chars, "", string)

def main():
    df = pd.read_csv(relative_path_1)
    reviews = df.loc[:, 'review'].values
    for ind, review in enumerate(reviews):
        if type(review) is float:
            reviews[ind] = ""

    reviews = clean_sentences("\n".join(reviews))
    with open(relative_path_2) as f:
        important_words = json.load(f)
    reviews = reviews.split("\n")
    n = len(reviews)
    d = len(important_words)
    X = np.zeros((n, d))
    y = df.loc[:, 'sentiment'].values
    y[y == -1] = 0

    for ind, review in enumerate(reviews):
        for ind_w, word in enumerate(important_words):
            X[ind, ind_w] = review.count(word)
    ones = np.ones((n, 1))
    X = np.concatenate((X, ones), axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    epochs = 50
    learning_rate = 0.1
    batch_size = 64
    logistic = LogisticRegression(epochs, learning_rate, batch_size)
    logistic.train(X_train, y_train)
    pred = logistic.predict(X_test)
    y_test = y_test.reshape((-1, 1))
    print("Accuracy: " + str(len(pred[pred == y_test])/len(pred)))

if __name__ == '__main__':
    main()