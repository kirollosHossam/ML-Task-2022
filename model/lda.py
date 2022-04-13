from preprocessing.preprocess import Preprocess
import gensim
from gensim import corpora, models
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS

class Lda():

    def __init__(self):
        self.lda_model_tfidf=gensim.models()

    def train(self):
        x=Preprocess()
        self.lda_model_tfidf = gensim.models.LdaMulticore(x.corpus_tfidf(), num_topics = 5, id2word = x.dictionary, passes = 10)
        for idx, topic in self.lda_model_tfidf.print_topics(-1):
            print("Topic: {} \nWord: {}".format(idx, topic))
            print("\n")

    def predict(self,documentnum):
        for index, score in sorted(self.lda_model_tfidf[x.bow_corpus()[documentnum]], key=lambda tup: -1 * tup[1]):
            print("\nScore: {}\t \nTopic: {}".format(score, self.lda_model_tfidf.print_topic(index, 10)))

    def test(self,unseen_document):
        bow_vector = x.dictionary.doc2bow(x.preprocess_vector(unseen_document))
        threshold = 0.1
        for index, score in sorted(self.lda_model_tfidf[bow_vector], key=lambda tup: -1 * tup[1]):
            if score < threshold: break
            print("Score: {}\n Topic: {}".format(score, self.lda_model_tfidf.print_topic(index, 5)))
            print()

    def word_cloud(topic, model):
        plt.figure(figsize=(8, 6))
        topic_words = [model.print_topic(topic, 75)]
        cloud = WordCloud(stopwords=STOPWORDS, background_color='white',
                          width=2500, height=1800).generate(" ".join(topic_words))

        print('\nWordcloud for topic:', topic, '\n')
        plt.imshow(cloud)
        plt.axis('off')
        plt.show()

    def visualize(self):
        for topic in range(10):
            plt.figure(figsize=(10, 15))
            self.word_cloud(topic, self.lda_model_tfidf)


if __name__ == '__main__':
    x=Lda()
    x.train()
    x.visualize()
