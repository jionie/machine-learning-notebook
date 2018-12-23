import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import imp
from sklearn.linear_model import Perceptron

if __name__ == "__main__":
    
    # load data
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['label'] = iris.target

    df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']

    #print(df)

    # plt.scatter(df[:50]['sepal length'], df[:50]['sepal width'], label='0')
    # plt.scatter(df[50:100]['sepal length'], df[50:100]['sepal width'], label='1')
    # plt.scatter(df[100:150]['sepal length'], df[100:150]['sepal width'], label='2')
    # plt.xlabel('sepal length')
    # plt.ylabel('sepal width')
    # plt.legend()
    # plt.show()

    data = np.array(df.iloc[:100, [0, 1, -1]])
    #select by df.iloc, select by index
    X = data[:, :-1]
    y = np.where(data[:, -1]==0, -1, data[:, -1])
    y = np.reshape(y, [y.shape[0], 1])

    #fit
    clf = Perceptron(fit_intercept=False, max_iter=1000, shuffle=False)
    clf.fit(X[:75], y[:75])

    #decision boundry, train with 75% points
    x_points = np.linspace(4, 7,10)
    y_ = -(clf.coef_[0][0]*x_points + clf.intercept_)/clf.coef_[0][1]
    plt.plot(x_points, y_)

    #data points
    plt.plot(data[:50, 0], data[:50, 1], 'bo', color='blue', label='0')
    plt.plot(data[50:100, 0], data[50:100, 1], 'bo', color='orange', label='1')
    plt.xlabel('sepal length')
    plt.ylabel('sepal width')
    plt.legend()
    plt.show()
