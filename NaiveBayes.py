'''
NaiveBayes.py
'''
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import MultinomialNB
from sklearn.datasets import make_blobs, make_classification
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt

'''
NB(X, y) fits data into the Gaussian Naive Bayes classifier and does grid search to find the best var_smoothing parameter for classification
:param X: n * m matrix (ndarray) containing vectors for classification
:param y: n * 1 matrix (ndarray) containing expected output for corresponding vector in X
:return: 
    gs_NB: the fitted NB model
    gs_NB.best_params_: Dictionary contatining parameter names and their corresponding best values
    accuracy_test: Testing accuracy of the model
'''



def NB(X, y):
    classifier = MultinomialNB()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    # scaler = MinMaxScaler()
    # X_train = scaler.fit_transform(X_train)
    # X_train = scaler.fit_transform(X_train)
    
    param_grid_nb = { 'alpha': np.logspace(0,-9, num=10) }
    
    multi_NB = GridSearchCV(estimator=classifier, param_grid=param_grid_nb, verbose=0,scoring='accuracy')
    
    multi_NB.fit(X_train, y_train)
 
    predict_test = multi_NB.predict(X_test)
    accuracy_test = accuracy_score(y_test,predict_test)

    return multi_NB, multi_NB.best_params_, accuracy_test

'''
Sample usecase, testing on classification dataset with 100000 samples and 50 features
'''
if __name__ == '__main__':
    X, y = make_classification(100000, n_features=50)
    print(X.shape, y.shape)
    print(X[0], y[0])
    
    model, params, score = NB(X, y)
    #returned model can be used to predict new data samples
    print("Prediction for", X[0], ":", model.predict(X[0].reshape(1, -1))[0])
    
    #Print params choosen by gridSearch
    print("Best Params:", params)

    #Print Score
    print("Test Accuracy:", score)
