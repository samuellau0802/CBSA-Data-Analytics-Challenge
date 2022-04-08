# import library

import pandas as pd
import re
from tqdm import tqdm
import numpy as np
import os
import jieba
from jieba.analyse.analyzer import ChineseAnalyzer
from jieba import tokenize

# dataframe to list

def load_data_from_pkl(df,x):
    '''
    Return list
    Load data from pickle. 'df' is dataframe. 'x' is column name
    '''
    headline = df.filter([x])
    headline = headline.to_numpy()
    headline = headline.tolist()

    return headline

# remove trash content

def remove_trash(temp):
    '''
    Return str
    Remove unwanted characters/sub strings
    '''
    temp = str(temp)                                    # ensure input is string type
    temp = re.sub(r'\n','',temp)                        # \n
    temp = re.sub(r'\\n','',temp)                       # \\n
    temp = re.sub(r'=====Shared Post=====','',temp)     
    temp = re.sub(r'http\S+', '', temp)                 # website
    temp = re.sub(r'/','',temp)                         # /
    temp = re.sub(r'＊','',temp)
    temp = re.sub(r'\[.*?\]','',temp)                   # emoji
    temp = re.sub(r"\u3000",'',temp)
    temp = re.sub(r'--','',temp)                        # consecutive -
    temp = re.sub(r"\*",'',temp)                        # *
    temp = re.sub(r'➤\S+','',temp)                     # ➤xxxx 
    temp = re.sub(r'\u200b','',temp)
    temp = re.sub(r'＝＝','',temp)
    temp = temp.lstrip()                                # beginning space
    return temp

def clean_content(df,col):
    '''
    Return dataframe
    'df' is dataframe. 'col' is target column.
    '''
    df = df[df['content'] != 'nan']
    #???
    df.dropna(subset = ['content'],inplace=True)
    df.reset_index(inplace=True)
    df[col] = df[col].apply(remove_trash)
    
    return df

def df_col_to_list(df,col):
    '''
    Return list
    'df' is dataframe. 'col' is column name
    '''
    df = df.filter([col])
    df = df.to_numpy()
    dflist = df.tolist()
    return dflist

def combine_result(cleanlist,original):
    '''
    Return dataframe
    Concatenate clean and original content as dataframe for later use
    'cleanlist' is list with clean content. 'original' is orginal list.
    '''
    result = pd.DataFrame({'Clean':cleanlist,'Original':original})
    return result

# save result
def save_as_pkl(name,df):
    '''
    No return
    Save file as name.pkl
    '''
    name = name + '.pkl'
    
    if os.path.exists(name):
        os.remove(name)

    df.to_pickle(name)
    
def preprocess(filename,col_name):
    '''
    No return
    Remove unwanted words and save as pkl
    '''
    df = clean_content(filename,col_name)
    df = df.filter([col_name])
    cleanlist = df_col_to_list(df,col_name)
    
    df2 = pd.read_pickle(filename)
    df2 = df2.filter([col_name])
    original = df_col_to_list(df2,col_name)
    
    result_df = combine_result(cleanlist,original)
    
    save_as_pkl('clean_'+col_name,result_df)

# Read and load clean data

def load_clean_data(filename):
    '''
    Return dataframe  
    'filename' is the file name of the clean pkl
    '''
    clean = pd.read_pickle(filename)
    
    return clean

# Tokenize

def tokenization(text):
    '''
    Return str
    '''
    text = re.sub(' ','',text)
    text = list(jieba.cut(text))
    text = [re.sub(r'[^\w]', '', i) for i in text if re.sub(r'[^\w]', '', i) != '']
    return ' '.join(text)

# Save tokenize result

def save_tokenized(name,df):
    '''
    No return
    Save tokenized result as name_tokenized.pkl
    '''
    if os.path.exists(name+'_tokenized.pkl'):
        os.remove(name+'_tokenized.pkl')
    resultdf = pd.DataFrame(df)
    resultdf.to_pickle(name+'_tokenized.pkl')

# Tokenizing

def tokenize(filename,col_name):
    '''
    Return dataframe
    'filename' is the clean file name. 'col_name' is the column name
    '''
    df = load_clean_data(filename)
    df[col_name] = df[col_name].dropna().astype(str)
    df[col_name] = df[col_name].apply(tokenization)
    df[col_name] = df[col_name].dropna().astype(str)
    
    save_tokenized(col_name,df)
    
    return df
