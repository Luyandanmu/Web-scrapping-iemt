#!/usr/bin/env python

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn import tree
import requests
from bs4 import BeautifulSoup

def main():
    iris = load_iris()
    X = iris.data
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy:.2f}')

    plt.figure(figsize=(10, 8))
    tree.plot_tree(clf, filled=True, feature_names=iris.feature_names, class_names=iris.target_names)
    plt.title('Decision Tree Visualization')
    plt.show()

if __name__ == '__main__':
    main()
