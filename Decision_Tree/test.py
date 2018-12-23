import numpy as np
from ID3 import *
from C4_5 import *
import pandas as pd
from sklearn.datasets import load_iris
from sklearn import tree 


if __name__ == "__main__":

    # load data
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['label'] = iris.target
    df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']
    
    data = np.asarray(df.iloc[:, :])
    np.random.shuffle(data)

    X = data[:, :-1]
    y = data[:, -1]

    #ID3 fit 
    tree_ID3 = ID3(X[:105, :], y[:105], 10000, 0)
    tree_ID3.fit(X[:105, :], y[:105])

    #ID3 test accuracy
    accuracy = tree_ID3.score(X[105:150], y[105:150])
    print("ID3 test accuracy:", accuracy)

    #C4.5 fit 
    tree_C4_5 = C4_5(X[:105, :], y[:105], 10000, 0)
    tree_C4_5.fit(X[:105, :], y[:105])

    #C4.5 test accuracy
    accuracy = tree_C4_5.score(X[105:150], y[105:150])
    print("C4.5 test accuracy:", accuracy)

    #sklearn decision tree, classification
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X[:105, :], y[:105])
    predict = clf.predict(X[105:150])
    result = np.where(predict==y[105:150], 1, 0)
    print("sklearn test accuracy:", np.sum(result)/result.shape[0])







    


