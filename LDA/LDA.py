import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import numpy as np

#LDA is supervised learning

def LDA(data):





if __name__ == "__main__":
    
    iris = datasets.load_iris()
    X = iris['data']
    y = iris['target']
    target_names = iris['target_names']