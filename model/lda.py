from preprocessing.preprocess import Preprocess
import gensim
from gensim import models
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from gensim.models import CoherenceModel 

class Lda():

    def __init__(self):
        self.lda_model_tfidf=gensim.models
        self.x=Preprocess()

    def train(self):
        self.lda_model_tfidf = gensim.models.LdaMulticore(self.x.corpus_tfidf(), num_topics = 7, id2word = self.x.dictionary, passes = 10)
        for idx, topic in self.lda_model_tfidf.print_topics(-1):
            print("Topic: {} \nWord: {}".format(idx, topic))
            print("\n")

    def evaluate(self):
        coherence_model_lda=CoherenceModel(model=self.lda_model_tfidf,texts=self.x.processed_df,dictionary=self.x.dictionary,coherence='c_v')
        coherece_lda=coherence_model_lda.get_coherence()
        print("Coherence Score = ",coherece_lda)

    def compute_coherence_values(self,limit,start=2,step=1):
        coherence_values=[]
        model_lists=[]
        for num_topics in range(start,limit,step):
            lda_model = gensim.models.LdaMulticore(self.x.corpus_tfidf(),chunksize=200, num_topics = num_topics, id2word = self.x.dictionary, passes = 10)
            model_lists.append(lda_model)
            coherence_model_lda = CoherenceModel(model=lda_model, texts=self.x.processed_df,
                                                 dictionary=self.x.dictionary, coherence='c_v')
            coherece_lda = coherence_model_lda.get_coherence()
            coherence_values.append((coherece_lda))
        return  model_lists,coherence_values

    def plot_coherence_values(self):
        model_list,coherence_values=self.compute_coherence_values(start=2,limit=10,step=1)
        x=range(2,10,1)
        plt.plot(x,coherence_values)
        plt.xlabel("Number of Topics")
        plt.ylabel("Coherence Score")
        plt.show()

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

    def word_cloud(self,topic, model):
        plt.figure(figsize=(8, 6))
        topic_words = [model.print_topic(topic, 75)]
        cloud = WordCloud(stopwords=STOPWORDS, background_color='white',
                          width=2500, height=1800).generate(" ".join(topic_words))

        print('\nWordcloud for topic:', topic, '\n')
        plt.imshow(cloud)
        plt.axis('off')
        plt.show()

    def visualize(self):
        for topic in range(7):
            plt.figure(figsize=(10, 15))
            self.word_cloud(topic, self.lda_model_tfidf)


# if __name__ == '__main__':
#     x=Lda()
#     x.train()
#     x.evaluate()
#     x.visualize()

