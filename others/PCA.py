'''
PCA.py
'''
from asyncio import base_tasks
import numpy as np
from sklearn.decomposition import PCA, IncrementalPCA


def Pca_M(X, m):
    '''
    PCA(X, m) takes dataset X and reduces it to a target dimension with the dimensions that explain the most variance
    :param X: dataset, ndarray of dimension n * a
    :param m: target dimension
    :output: Shape of transformed dataset and explained variance ratio of each dimension
    :return: transformed dataset, ndarray of dimension n * m
    '''
    pca = PCA(n_components=m)
    red_X = pca.fit_transform(X)
    print("Shape of X:", red_X.shape)
    print("Explained variance ratio: ", pca.explained_variance_ratio_)
    return red_X

def Pca_T(X):
    '''
    PCA(X) takes dataset X and reduces it to dimensions such that at least 85% of variance is explained
    :param X: dataset, ndarray of dimension n * a
    :output: Shape of transformed dataset and explained variance ratio of each dimension
    :return: transformed dataset, ndarray of dimension n * m
    '''
    pca = PCA(n_components=min(X.shape[1], 100))
    red_X = pca.fit_transform(X)
    i = 0
    sum = 0
    while sum < 0.85:
        sum += pca.explained_variance_ratio_[i]
        i += 1
    red_X = red_X[:,[0,i]]
    print("Shape of X:", red_X.shape[:i])
    print("Explained variance ratio: ", pca.explained_variance_ratio_[:i])
    return red_X

def iPca(X, m, b):
    '''
    iPCA(X, m) performs PCA in batch with batchsize = b
    :param X: dataset, ndarray of dimension n * a
    :param m: target dimension
    :param b: batch size
    :output: Shape of transformed dataset and explained variance ratio of each dimension
    :return: transformed dataset, ndarray of dimension n * m
    '''
    pca = IncrementalPCA(n_components=m, batch_size=b)
    red_X = pca.fit_transform(X)
    print("Shape of X:", red_X.shape)
    print("Explained variance ratio: ", pca.explained_variance_ratio_)
    return red_X

if __name__ == "__main__":
    test_X = np.array([[-1, -1, 3, 0], [-2, -1, -4, 2], [-3, -2, 0, 1], [1, 1, 1, 1], [2, 1, 2, 0], [3, 2, 5, 2]])
    test_m = 3
    new_X = Pca_T(test_X)
    print(new_X)