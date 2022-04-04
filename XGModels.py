from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score, RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, mean_absolute_error
from sklearn.datasets import make_classification, make_regression

def XG_preprocess(X):
    pre_pipeline = Pipeline(
        steps=[("impute", SimpleImputer(strategy="mean")), 
           ("scale", StandardScaler(with_mean=False))]
    )

    X_processed = pre_pipeline.fit_transform(X)
    return X_processed

def XG_Classifier(X, y):
    '''
    Preprocesses data and train XGBClassifer on dataset
    :param X: n * m matrix (ndarray) containing vectors for classification
    :param y: n * 1 matrix (ndarray) containing expected binary or multiclass output for corresponding vector in X
    :return: 
        xgbclassify: fitted XGBClassifer model object
        score: f1_score on testing set    
    '''
    X_processed = XG_preprocess(X)
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, stratify=y)

    xgbclassify = xgb.XGBClassifier()
    xgbclassify.fit(X_train, y_train)
    preds = xgbclassify.predict(X_test)

    #score = f1_score(y_test, preds)
    return xgbclassify, preds, y_test

def XG_Regression(X,y):
    '''
    Train XGBRegressor on dataset
    :param X: n * m matrix (ndarray) containing vectors for classification
    :param y: n * 1 matrix (ndarray) containing expected continuous output for corresponding vector in X
    :return: 
        xgbregression: fitted XGBRegressor model object
        score: Mean absolute error on testing set    
    '''
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    xgbregression = xgb.XGBRegressor()

    xgbregression.fit(X_train, y_train)
    preds = xgbregression.predict(X_test)
    score = mean_absolute_error(y_test, preds)

    return xgbregression, score

if __name__ == "__main__":
    X, y = make_classification(n_samples = 100000)
    print(X[0], y[0])
    model, score = XG_Classifier(X,y)
    test_pred = model.predict(X[0].reshape(1, -1))
    print("Prediction:", test_pred)
    print("f1_score", score)

    


