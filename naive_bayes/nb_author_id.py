#!/usr/bin/python

"""
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project.

    Use a Naive Bayes Classifier to identify emails by their authors

    authors and labels:
    Sara has label 0
    Chris has label 1
"""

import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn.naive_bayes import GaussianNB


def train(features_train, labels_train):
    t0 = time()
    clf = GaussianNB()
    clf.fit(features_train, labels_train)
    print "training time:", round(time()-t0, 3), "s"
    return clf


def test(clf, features_test, labels_test):
    return clf.score(features_test, labels_test)


def main():
    ### features_train and features_test are the features for the training
    ### and testing datasets, respectively
    ### labels_train and labels_test are the corresponding item labels
    features_train, features_test, labels_train, labels_test = preprocess()
    print("Start Training")
    clf = train(features_train, labels_train)
    print("End of training!")

    ## Check for accuracy of the classifier
    print(test(clf, features_test, labels_test))


if __name__ == '__main__':
    main()
