import pandas as pd
import numpy as np
import gensim.corpora as corpora
from gensim.models import LdaMulticore as LDA
from gensim.models.coherencemodel import CoherenceModel
import pyLDAvis.gensim_models
import matplotlib.pyplot as plt

DOC_CONTENT = 'Clean'
DOC_ID = 'docid'



def formatting(data):
    data['Clean'] = data['Clean'].apply(filter_sw)
    seg_list = data['Clean'].tolist()
    return seg_list

def genCorpora(seg_list):
    id2word = corpora.Dictionary(seg_list)
    corpus = [id2word.doc2bow(text) for text in seg_list]

    return id2word, corpus

def trainLDA(id2word, corpus, num_topics, texts):
    lda_model = LDA(corpus=corpus, id2word=id2word, num_topics=num_topics, workers=6)
    cv = CoherenceModel(model=lda_model, corpus=corpus, texts=texts, coherence='u_mass').get_coherence()

    return lda_model, cv

def optim_LDA(id2word, corpus, texts):
    model_list = []
    cv_list = []

    for num_topics in range(3, 16, 3):
        model, cv = trainLDA(id2word=id2word, corpus=corpus, num_topics=num_topics, texts=texts)
        model_list.append(model)
        cv_list.append(cv)

    plt.figure()
    plt.plot([i for i in range(3, 16, 3)], cv_list)
    plt.show()

    return model_list, cv_list
    
def save_model(model, corpus):
    model.save('lda_optim.model')

    vis = pyLDAvis.gensim_models.prepare(model, corpus, dictionary=model.id2word)
    with open("lda_results.html", "w") as file:
        file.write(pyLDAvis.display(vis).data)