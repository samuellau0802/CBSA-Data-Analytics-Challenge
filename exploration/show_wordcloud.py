''''
Require SourceHanSerifK-Light.otf to display chinese words
'''
from wordcloud import WordCloud
import matplotlib.pyplot as plt

with open ("stopwords.txt", "r") as stopwords:
    stopwords = stopwords.read().splitlines()


def show_WC(df, label, max_words=1000):
    '''given a dataframe that contains tokenized words and the label, return a wordcloud image
    param: df(dataframe): a dataframe that contains column tokenized_words and label
    param: label(str/int): the label name that would like to filter
    param: max_words(int): number of words would like to show in the wordcloud
    return: an image of wordcloud 
    '''

    wc = WordCloud(font_path="font\SourceHanSerifK-Light.otf", background_color="white", max_words=max_words,
               max_font_size=300, random_state=42, width=1000, height=860, margin=2, stopwords=set(stopwords))
    
    words = ' '.join(list(df[df['label'] == label]['tokenized_words']))

    wc.generate(words)

    plt.figure(figsize=(15,15))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    return plt.show()

