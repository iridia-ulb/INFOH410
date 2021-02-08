#!/usr/bin/python3

from sklearn import tree
import pandas
from matplotlib import pyplot as plt

# For this TP you need sklearn, pandas, and matplotlib
# e.g. pip install sklearn pandas matplotlib


def q5():
    """
    This function should load the dataset and then use pandas and sklearn
    to grow a decision tree.
    For processing the data see: https://scikit-learn.org/stable/modules/classes.html#module-sklearn.preprocessing
    and https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.get_dummies.html
    for decision trees:
    https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
    """
    df = pandas.read_csv("dataset1.csv")
    print(df)
    # ...
    tree.plot_tree(clf, filled=True, feature_names=X.columns)
    plt.show()


if __name__ == "__main__":
    q5()
