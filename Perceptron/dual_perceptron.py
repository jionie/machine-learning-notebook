import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import imp

# $f(x) = sign(\sum_{j=1}^{N}\alpha_{j}x_{j}x+b)$
# 损失函数 $L(w, b) = -\ {y_{i}(w*x_{i} + b)}$
# if y_{i}(\sum_{j=1}^{N}\alpha_{j}x_{j}x_{i}+b) <=0 
# update by \alpha_{i} += lr, b += lr*y_{i}
#w = \sum_{i=1}^{N}\alpha_{i}x_{i}y_{i}
#b = \sum_{i=1}^{N}\alpha_{i}y_{i}

class Perceptron:
    def __init__(self, X_train, y_train):
    
        self.Gram_matrix = np.dot(X_train, np.transpose(X_train))
        self.alpha = np.zeros(X_train.shape[0])
        self.w = np.ones(X_train.shape[1])
        self.b = np.zeros(y_train.shape[1])
        self.lr = 0.1

    def predict(self, x): 
        y = np.dot(x, self.w) + self.b
        return y

    def fit(self, X_train, y_train):

        is_wrong = False
        while not is_wrong:
            wrong_count = 0
            for i in range(X_train.shape[0]):
                sign = y_train[i]*\
                (np.sum(np.multiply(self.Gram_matrix[i], \
                np.multiply(np.transpose(y_train), \
                np.reshape(self.alpha, [1, X_train.shape[0]]))))+self.b)
                if sign <= 0:
                    self.alpha[i] += self.lr
                    self.b += self.lr*y_train[i]
                    wrong_count += 1

            if(wrong_count==0):
                is_wrong = True
        y_train_ = np.repeat(y_train, X_train.shape[1]).reshape([X_train.shape[0], X_train.shape[1]])
        alpha_ = np.repeat(self.alpha, X_train.shape[1]).reshape([X_train.shape[0], X_train.shape[1]])
      
        self.w = np.sum(np.multiply(alpha_, np.multiply(y_train_, X_train)), axis=0)
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
    perceptron = Perceptron(X[:75], y[:75])
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