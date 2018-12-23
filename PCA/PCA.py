import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA as PCA_sklearn
import numpy as np

def PCA(data, n_components):

    #first compute the mean of every feature
    mean = np.mean(data, axis=0)

    #then shift the data
    shifted_data = data - mean

    #then compute the covariance matrix
    cov = np.cov(shifted_data, rowvar=False) 

    #then compute the eigenvalues
    eigenValues, eigVectors = np.linalg.eig(cov)

    #then sort the eigenvalues and eigenvectors by eigenvalues
    #numpy.argsort(a, axis=-1, kind='quicksort', order=None)
    #return the indices that would sort an array, smallest to largest
    eigenValInd = np.argsort(eigenValues)
    eigenValInd = eigenValInd[:-(n_components+1):-1]

    eigenValues = eigenValues[eigenValInd]
    eigVectors = eigVectors[:, eigenValInd]

    #transform shifted data to low dimenssion
    low_dimenssion_data = np.dot(shifted_data, eigVectors)

    #reconstructed data
    reconstructed_data = np.dot(low_dimenssion_data, eigVectors.T)+mean

    return low_dimenssion_data, reconstructed_data

if __name__ == "__main__":

    iris = datasets.load_iris()
    X = iris['data']
    y = iris['target']
    target_names = iris['target_names']
    X_r, X_rc = PCA(X, n_components=2)
    pca = PCA_sklearn(n_components=2)
    X_r_sklearn = pca.fit(X).transform(X)

    print(X_r, X_r_sklearn)

    f, axarr = plt.subplots(2, sharex=True)

    colors = ['navy', 'turquoise', 'darkorange']
    lw = 2

    for color, i, target_name in zip(colors, [0, 1, 2], target_names):
        axarr[0].scatter(X_r[y == i, 0], X_r[y == i, 1], color=color, alpha=.8, lw=lw,
                    label=target_name)
    axarr[0].legend(loc='best', shadow=False, scatterpoints=1)
    axarr[0].set_title('PCA of IRIS dataset')


    for color, i, target_name in zip(colors, [0, 1, 2], target_names):
        axarr[1].scatter(X_r_sklearn[y == i, 0], X_r_sklearn[y == i, 1], color=color, alpha=.8, lw=lw,
                    label=target_name)
    axarr[1].set_title('sklearn PCA of IRIS dataset')


    plt.tight_layout()
    plt.show()