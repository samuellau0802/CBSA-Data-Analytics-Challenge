from sklearn.feature_extraction.text import TfidfVectorizer

def tfidf(df, max_df=1.0, min_df=1):
    '''
    tfidf converts the dataframe with tokenized text into a matrix of numeric representation of the documents

    :param df: a dataframe that contains tokenized_text columns, with n rows
    :param max_df: sklearn tfidf parameters. default=1.0. ignore terms that have a document frequency strictly higher than the given threshold
    :param min_df: sklearn tfidf parameters. default=1. ignore terms that have a document frequency strictly lower than the given threshold

    :return a nxb matrix, where b equals to the number of unique vocabulary
    '''

    vectorizer = TfidfVectorizer(max_df=max_df, min_df=min_df)
    X = vectorizer.fit_transform(list(df['tokenized_text']))
    X = X.todense()
    return X, vectorizer.vocabulary_