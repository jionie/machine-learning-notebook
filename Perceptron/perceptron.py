import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import imp

# $f(x) = sign(w*x + b)$
# 损失函数 $L(w, b) = -\Sigma{y_{i}(w*x_{i} + b)}$
# update by w += lr*y_{i}*x_{i}, b += lr*y_{i}

class Perceptron:
    def __init__(self):
        self.w = np.ones([1])
        self.b = np.zeros([1])
        self.lr = 0.1

    def predict(self, x):
        y = np.dot(x, self.w) + self.b
        return y

    def fit(self, X_train, y_train):
        self.w = np.ones(X_train.shape[1])
        self.b = np.zeros(y_train.shape[1])
        is_wrong = False
        while not is_wrong:
            wrong_count = 0
            for i in range(X_train.shape[0]):
                if y_train[i]*self.predict(X_train[i]) <= 0:
                    self.w += self.lr*y_train[i]*X_train[i]
                    self.b += self.lr*y_train[i]
                    wrong_count += 1

            if(wrong_count==0):
                is_wrong = True

        return "Fit successfully"

    def score(self, X_test, y_test):
        worng_count = 0
        for i in range(X_test.shape[0]):
            if(y_test[i]*self.predict(X_test[i])) <= 0:
                worng_count += 1
        
        return 1-worng_count/X_test.shape[0]



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
    perceptron = Perceptron()
    perceptron.fit(X[:75], y[:75])

    #test accuracy
    accuracy = perceptron.score(X[75:100], y[75:100])
    print("test accuracy:", accuracy)

    #decision boundry, train with 75% points
    x_points = np.linspace(4, 7,10)
    y_ = -(perceptron.w[0]*x_points + perceptron.b)/perceptron.w[1]
    plt.plot(x_points, y_)

    #data points
    plt.plot(data[:50, 0], data[:50, 1], 'bo', color='blue', label='0')
    plt.plot(data[50:100, 0], data[50:100, 1], 'bo', color='orange', label='1')
    plt.xlabel('sepal length')
    plt.ylabel('sepal width')
    plt.legend()
    plt.show()




