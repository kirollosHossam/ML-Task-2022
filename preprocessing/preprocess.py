import pandas as pd
import gensim
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from gensim import corpora, models
import numpy as np
np.random.seed(400)

class Preprocess():

    def __init__(self):
        self.dictionary=gensim.corpora.Dictionary()
        self.df=pd.DataFrame()
        self.processed_df=pd.DataFrame()

    def read_csv(self):
        self.df = pd.read_csv("F:\AI ITI\projects\ML Task\data\Pubmed5k.csv")
        # print(self.df.shape)
        self.df=self.df[self.df['Abstract'].map(len) > 50]
        # print(self.df.shape)
        self.df["text"]=self.df["Abstract"]+self.df["Title"]
        self.df.drop(['Abstract','Title','ArticleID'], axis=1, inplace=True)
        return self.df

    def stemming(self,text):
        stemmer = SnowballStemmer("english")
        return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

    def lemmatize(self,text):
        result = []
        for token in gensim.utils.simple_preprocess(text):
            if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) >= 2:
                result.append(self.stemming(token))
        return result

    def apply_preprocess(self):
        self.df=self.read_csv()
        self.processed_df=self.df['text'].map(lambda x: self.lemmatize(x))
        # print(self.processed_df.head())

    def preprocess_vector(self,text):
        return self.lemmatize(text)

    def get_dictionary(self):
        self.apply_preprocess()
        self.dictionary=gensim.corpora.Dictionary(self.processed_df)

    def filter_extremes(self):
        self.get_dictionary()
        # print(len(self.dictionary))
        # from collections import Counter
        # count = Counter()
        # for doc in self.apply_preprocess():
        #     for word in doc:
        #         count[word] += 1
        # print(count)
        self.dictionary.filter_extremes(no_below=5, no_above=0.1, keep_n=100000)
        # print(len(self.dictionary))


    def bow_corpus(self):
        self.filter_extremes()
        bow=[self.dictionary.doc2bow(doc) for doc in self.processed_df]
        # print(bow[0])
        return bow

    # def tf_idf(self):
    #     x=self.bow_corpus()
    #     return models.TfidfModel(x)

    def corpus_tfidf(self):
        x=self.bow_corpus()
        return models.TfidfModel(x)[x]


