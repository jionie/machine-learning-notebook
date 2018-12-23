from decision_tree import DecisionTree
import csv
import numpy as np  # http://www.numpy.org
import ast

X = list()
y = list()
XX = list()
numerical_cols = set([0,10,11,12,13,15,16,17,18,19,20])
with open("hw4-data.csv") as f:
    next(f, None)
    
    for line in csv.reader(f, delimiter=","):
        xline = []
        for i in range(len(line)):
            if i in numerical_cols:
                xline.append(ast.literal_eval(line[i]))
            else:
                xline.append(line[i])

        X.append(xline[:-1])
        y.append(xline[-1])
        XX.append(xline[:])

y_ = []
for record in X:
    if record in X:
        index = X.index(record)
        print(y[index])
        y_ = np.append(y_, y[index])

print(y_)