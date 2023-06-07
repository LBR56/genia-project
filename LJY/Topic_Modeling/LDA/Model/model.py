from gensim.models.ldamodel import LdaModel
from gensim.models import CoherenceModel
from gensim.models.callbacks import CoherenceMetric
from gensim.models.callbacks import PerplexityMetric
import numpy as np
import matplotlib.pyplot as plt
from Preprocess.text_preprocess import Preprocess


class LDAModeling:
    """
    Class description : LDA 모델링 실험을 통해 실험에서 발생하는 모델 리스트와 각 지표들을 반환하고, 이에 관해 plot을 보여줍니다.
    
    Params
        texts : Preprocessed texts
        dictionary : Gensim dictionary
        corpus : Gensim corpus
        limit : Limit number of topcis
        start : start number of topics
        step : step for experiments
    """

    def __init__(self, texts, dictionary, corpus, limit, passes=10, start=2, step=3):
        self.texts = texts
        self.dictionary = dictionary
        self.corpus = corpus
        self.limit = limit
        self.start = start
        self.step = step
        self.passes = passes

    def compute_evaluation_index(self):
        """
        Func description : LDA 모델을 평가하기 위해 토픽 수 별 평가 지표를 반환 합니다

        Returns
            model_list : list of models conducted experiments
            perplexity_values : 토픽 수 별 LDA 모델의 perplexity values를 반환합니다
            coherence_values : 토픽 수 별 LDA 모델의 coherence values를 반환합니다
        """
        coherence_values = list()
        perplexity_values = list()
        model_list = list()
        for num_topics in range(self.start, self.limit, self.step):
            model = LdaModel(corpus=self.corpus, id2word=self.dictionary, num_topics=num_topics, passes=self.passes, random_state=42)
            coherencemodel = CoherenceModel(model=model, texts=self.texts, dictionary=self.dictionary, coherence='c_v')
            perplexity_values.append(model.log_perplexity(self.corpus))
            coherence_values.append(coherencemodel.get_coherence())
            model_list.append(model)

        return model_list, coherence_values, perplexity_values


    def plot_optimal_number_of_topics(self, save=False):
        """
        Func description : coherence values & perplexity values에 따라 최적화된 토픽 수를 찾기 위해 plotting


        Returns
            pyplot show for graphs of each indices
        """

        model_list, coherence_values, perplexity_values = self.compute_evaluation_index()

        x = range(self.start, self.limit, self.step)

        plt.subplot(2,1,1)
        plt.plot(x, coherence_values, '.-r')
        plt.xlabel("Num Topics")
        plt.ylabel("Coherence score")
        plt.legend(("coherence_values"), loc='best')

        plt.subplot(2,1,2)
        plt.plot(x, perplexity_values, '.-g')
        plt.ylabel("perplexity score")
        plt.legend(("perplexity_values"), loc='best')

        plt.tight_layout()

        if save == True:
            plt.savefig("./Data/png/Cohe_perpl_plot.png")
            
        plt.show()

    def get_lda_model(self, num_of_topics):
        ldamodel = LdaModel(
            corpus=self.corpus,
            id2word=self.dictionary,
            num_topics=num_of_topics,
            passes=self.passes,
            random_state=42
        )

        return ldamodel


            