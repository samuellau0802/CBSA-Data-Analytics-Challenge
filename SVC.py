'''
SVC.py
'''
from sklearn.svm import SVC 
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.datasets import make_blobs, make_classification
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt

'''
svc(X, y) fits data into the Support Vector Classifier and does grid search to find the best var_smoothing parameter for classification
:param X: n * m matrix (ndarray) containing vectors for classification
:param y: n * 1 matrix (ndarray) containing expected output for corresponding vector in X
:return: 
    svc_model: the fitted SVC model
    svc_model.best_params_: Dictionary contatining parameter names and their corresponding best values
    accuracy_test: Testing accuracy of the model
'''
def svc(X, y):
    classifier = SVC()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    param_grid_svc = {
        'C': [1, 10, 100, 1000], 
        'gamma': [0.001, 0.0001], 
        'kernel': ['rbf']},
    
    svc_model = GridSearchCV(estimator=classifier, param_grid=param_grid_svc, verbose=1,scoring='accuracy')
    svc_model.fit(X_train, y_train)
    predict_test = svc_model.predict(X_test)
    accuracy_test = accuracy_score(y_test,predict_test)
    return svc_model, svc_model.best_params_, accuracy_test


'''
Sample usecase, testing on classification dataset with 10000 samples and 50 features
'''
if __name__ == '__main__':
    X, y = make_classification(10000, n_features=50)
    print(X.shape, y.shape)
    print(X[0], y[0])
    
    model, params, score = svc(X, y)
    #returned model can be used to predict new data samples
    print("Prediction for", X[0], ":", model.predict(X[0].reshape(1, -1))[0])
    
    #Print params choosen by gridSearch
    print("Best Params:", params)

    #Print Score
    print("Test Accuracy:", score)
