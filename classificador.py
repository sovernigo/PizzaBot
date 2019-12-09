from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
import csv


import pandas as pd
import numpy as np

modelScore = dict()

vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, strip_accents='unicode')


def getData():
    vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, strip_accents='unicode')

    data = csv.reader(open('intencoes.csv', 'r'))
    corpus, y = [], []
    for row in data:
        corpus.append(row[1])
        y.append(row[0])

    x = vectorizer.fit_transform(corpus)

    return x, y


def knn(n):

    x, y = getData()

    model = KNeighborsClassifier(n_neighbors=n)

    loo = LeaveOneOut()
    loo.get_n_splits(x)
    y_pred = []
    a = np.array(y)
    for train_index, test_index in loo.split(x):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = a[train_index], a[test_index]

        model.fit(x_train, y_train)
        y_pred.extend(model.predict(x_test))


    print("KNN com N =", n)
    print("y: ")
    print(y)
    print("y_pred: ")
    print(y_pred)

    print("recall score:")
    print("macro:", recall_score(y, y_pred, average='macro'))

    print("micro:", recall_score(y, y_pred, average='micro'))

    print(recall_score(y, y_pred, average=None))

    print("precision score:")
    print("macro:", precision_score(y, y_pred, average='macro'))

    print("micro", precision_score(y, y_pred, average='micro'))

    print("weighted:", precision_score(y, y_pred, average='weighted'))

    print(precision_score(y, y_pred, average=None))

    print("accuracy score:")
    print("normalizado:", accuracy_score(y, y_pred))

    print("nao normalizado:", accuracy_score(y, y_pred, normalize=False), "\n")


def DecisionTree():
    print("Decision Tree")
    x, y = getData()
    clf = DecisionTreeClassifier()

    #tree.plot_tree(clf.fit(x, y))

    loo = LeaveOneOut()
    loo.get_n_splits(x)
    y_pred = []
    a = np.array(y)
    for train_index, test_index in loo.split(x):
      x_train, x_test = x[train_index], x[test_index]
      y_train, y_test = a[train_index], a[test_index]

      clf.fit(x_train, y_train)
      y_pred.extend(clf.predict(x_test))

    print(y)
    print(y_pred)
    print("recall score:")
    print("macro:", recall_score(y, y_pred, average='macro')  )

    print("micro:", recall_score(y, y_pred, average='micro')  )

    print(recall_score(y, y_pred, average=None))

    print("precision score:")
    print("macro:", precision_score(y, y_pred, average='macro')  )

    print("micro", precision_score(y, y_pred, average='micro')  )

    print("weighted:", precision_score(y, y_pred, average='weighted'))

    print(precision_score(y, y_pred, average=None) )

    print("accuracy score:")
    print("normalizado:", accuracy_score(y, y_pred))

    print("nao normalizado:", accuracy_score(y, y_pred, normalize=False), "\n")

def LogisticReg():
    print('LogisticRegression')
    x, y = getData()

    modelLogistic = LogisticRegression()

    loo = LeaveOneOut()
    loo.get_n_splits(x)
    y_pred = []
    a = np.array(y)
    for train_index, test_index in loo.split(x):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = a[train_index], a[test_index]

        modelLogistic.fit(x_train, y_train)
        y_pred.extend(modelLogistic.predict(x_test))

    print(y)
    print(y_pred)
    print("recall score:")
    print("macro:", recall_score(y, y_pred, average='macro'))

    print("micro:", recall_score(y, y_pred, average='micro'))

    print(recall_score(y, y_pred, average=None))

    print("precision score:")
    print("macro:", precision_score(y, y_pred, average='macro'))

    print("micro", precision_score(y, y_pred, average='micro'))

    print("weighted:", precision_score(y, y_pred, average='weighted'))

    print(precision_score(y, y_pred, average=None))

    print("accuracy score:")
    print("normalizado:", accuracy_score(y, y_pred))

    print("nao normalizado:", accuracy_score(y, y_pred, normalize=False), "\n")

#KNN com n = 1, 3 e 5
knn(1)

knn(3)

knn(5)

DecisionTree()

LogisticReg()