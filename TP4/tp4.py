#!/usr/bin/python3

from sklearn import tree
import pandas
import numpy as np
from matplotlib import pyplot as plt


def main():
    df = pandas.read_csv("dataset1.csv")
    print(df)
    clf = tree.DecisionTreeClassifier(criterion="entropy")
    X = pandas.get_dummies(df.iloc[:, :-1])
    y = df.sunburn
    print(X)
    print(y)
    clf.fit(X, y)
    tree.plot_tree(clf, filled=True, feature_names=X.columns)
    plt.show()


if __name__ == "__main__":
    main()
