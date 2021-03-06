from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
import pandas as pd

def tfidf(df, max_df=1.0, min_df=1):
    '''
    tfidf converts the dataframe with tokenized text into a matrix of numeric representation of the documents

    :param df: a dataframe that contains tokenized_text columns, with n rows
    :param max_df: sklearn tfidf parameters. default=1.0. ignore terms that have a document frequency strictly higher than the given threshold
    :param min_df: sklearn tfidf parameters. default=1. ignore terms that have a document frequency strictly lower than the given threshold

    :return a n x b matrix, where b equals to the number of unique vocabulary
    :return a dictionary of vocabulary that maps the terms to feature indices.
    '''

    vectorizer = TfidfVectorizer(max_df=max_df, min_df=min_df)
    X = vectorizer.fit_transform(list(df['tokenized_text']))
    #X = X.todense() # sprase to dense matrix
    return X, vectorizer.vocabulary_

def transformer(df, model_name='paraphrase-multilingual-mpnet-base-v2'):
    '''
    sentence transformer converts the dataframe with raw text into a matrix of word embeddings. supports GPU Encoding

    :param df: a dataframe that contains text columns, with n rows / a series of strings
    :param model: pretrained model that you would like to use

    :return a n x 728 matrix
    '''
    model = SentenceTransformer(model_name, device='cuda')
    if isinstance(df, pd.Series):
        embeddings = model.encode(list(df), convert_to_tensor=True)
    else:
        embeddings = model.encode(list(df['text']), convert_to_tensor=True)

    return embeddings