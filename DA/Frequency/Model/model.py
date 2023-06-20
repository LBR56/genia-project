from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from Preprocess.preprocess import tokenizer
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from gensim.models.doc2vec import TaggedDocument
from gensim.models import doc2vec
from tqdm.notebook import tqdm
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix



class TfIdf:
    def __init__(self, texts, tokenizer):
        self.texts = texts
        self.tokenizer = tokenizer


    def transform_into_vector(self, popular_list, unpopular_list, min_df=3):
        self.tfidf_vectorizer = TfidfVectorizer(
            min_df=min_df,
            tokenizer=self.tokenizer
        ).fit(self.texts)

        popular_tfidf = self.tfidf_vectorizer.transform(popular_list).toarray()
        unpopular_tfidf = self.tfidf_vectorizer.transform(unpopular_list).toarray()

        return popular_tfidf, unpopular_tfidf
    
    def get_keywords(self, popular_tfidf, unpopular_tfidf, save=False):
        

        invert_index_vectorizer = {v: k for k, v in self.tfidf_vectorizer.vocabulary_.items()}

        po_keyword_list = list()
        unpo_keyword_list = list()
        for idx in range(len(popular_tfidf)):
            po_keywords_idx = [i for i in np.argsort(popular_tfidf[idx])[-5:]]
            unpo_keywords_idx = [i for i in np.argsort(unpopular_tfidf[idx])[-5:]]
            po_keywords = [invert_index_vectorizer[index] for index in po_keywords_idx]
            unpo_keywords = [invert_index_vectorizer[index] for index in unpo_keywords_idx]
            po_keyword_list.append(po_keywords)
            unpo_keyword_list.append(unpo_keywords)

        po_kw_df = pd.DataFrame(po_keyword_list)
        unpo_kw_df = pd.DataFrame(unpo_keyword_list)

        if save==True:
            po_kw_df.to_csv("../Data/csv/po_kw_df.csv")
            unpo_kw_df.to_csv("../Data/csv/unpo_kw_df.csv")

        return po_kw_df, unpo_kw_df


    @staticmethod
    def _save_cosine_mtx(popular_tfidf, unpopular_tfidf):
        cos_sim_df = pd.DataFrame(cosine_similarity(popular_tfidf, unpopular_tfidf))
        print(f"TF-IDF 변환 후 유사도 비교 시, 유사도의 평균은 {round(np.mean(cos_sim_df.values), 3)} 입니다.")
        cos_sim_df.to_csv("..Data/csv/cos_sim_df.csv")
        return cos_sim_df
    
    @staticmethod
    def _save_masked_heatmap(cos_sim_df):
        corr = cos_sim_df
        mask = corr.abs() < 0.5
        f, ax = plt.subplots(figsize=(11, 9))

        sns.heatmap(corr, mask=mask, cmap="Reds", vmax=1, center=0,
                    square=True, linewidths=.5, cbar_kws={"shrink": .5})
        
        plt.savefig("../Data/png/cosine_sim.png")



class Doc2Vec:
    def __init__(self, popular_list, unpopular_list):
        self.popular_list = popular_list
        self.unpopular_list = unpopular_list
        self.tokenizer = tokenizer

    def get_doc2vec_sets(self):
        popular_doc_df = pd.DataFrame({"text" : self.popular_list})
        unpopular_doc_df = pd.DataFrame({"text" : self.unpopular_list})

        total_doc_df = pd.concat([popular_doc_df, unpopular_doc_df], ignore_index=True)

        total_doc_df["tag"] = [f"인기강좌{i}" for i in range(1,31)] + [f"비인기강좌{i}" for i in range(1,31)]
        
        tagged_corpus_list = []
        for _, row in tqdm(total_doc_df.iterrows(), total=len(total_doc_df)):
            text = row['text']
            tag = row['tag']
            tagged_corpus_list.append(TaggedDocument(tags=[tag], words=tokenizer(text, mode="morphs")))

        return total_doc_df, tagged_corpus_list
    
    def fit(self, tagged_corpus_list, vector_size=300, alpha=0.025, min_alpha=0.025, workers=8, window=8):
        self.model = doc2vec.Doc2Vec(
            vector_size=vector_size,
            alpha=alpha,
            min_alpha=min_alpha,
            workers=workers,
            window=window,
        )

        self.model.build_vocab(tagged_corpus_list)
        self.model.train(tagged_corpus_list, total_examples=self.model.corpus_count, epochs=50)


    def get_similarity_rate(self, save=False):
        rates_po = []
        for i in range(1, 31):    
            rates_po.append(len([k for k , v in self.model.dv.most_similar(f'인기강좌{i}') if "비인기강좌" not in k])/10)

        rates_unpo = []
        for i in range(1, 31):    
            rates_unpo.append(len([k for k , v in self.model.dv.most_similar(f'비인기강좌{i}') if "비인기강좌" in k])/10)
    
        print(f"인기 강좌 별 유사강의 TOP 10에서 '인기강좌'인 비율의 평균은 {round(np.mean(rates_po) ,4) * 100 }% 입니다")
        print(f"비인기 강좌 별 유사강의 TOP 10에서 '비인기강좌'인 비율의 평균은 {round(np.mean(rates_unpo) ,4) * 100 }% 입니다")

        plt.figure(figsize=(8,6))
        plt.subplot(1,2,1)
        sns.histplot(rates_po)
        plt.title("popular among popular lecs")
        plt.xlabel("rates")

        plt.subplot(1,2,2)
        sns.histplot(rates_unpo)
        plt.title("unpopular among unpopular lecs")
        plt.xlabel("rates")

        plt.tight_layout()

        if save == True:
            plt.savefig("../Data/png/lec_similarity_rates.png")

        plt.show()



class PcaCluster:
    def __init__(self, pca_n_components, random_state):
        self.pca_n_components = pca_n_components
        self.random_state = random_state


    def get_df_pca(self, vectors):
        self.feat_cols = ["Doc" + str(i) for i in range(vectors.shape[1])]

        df = pd.DataFrame(vectors, columns=self.feat_cols)
        df["label"] = ["0"]* int(len(df)/2) + ["1"]* int(len(df)/2)

        return df


    def fit_transform(self, df):
        new_df = df.copy()
        self.pca = PCA(n_components=self.pca_n_components, random_state=self.random_state)
        pca_result = self.pca.fit_transform(new_df[self.feat_cols].values)

        new_df["pca-one"] = pca_result[:, 0]
        new_df["pca-two"] = pca_result[:, 1]
        new_df["pca-three"] = pca_result[:, 2]

        print("Explained variation per principle component: {}".format(self.pca.explained_variance_ratio_))

        return new_df
    
    @staticmethod
    def _plot_pca_results(df, dim, save=False):
        try:
            if dim != 2 and dim != 3:
                raise Exception("The dim should be 2 or 3")

            if dim == 2:
                plt.figure(figsize=(10, 6))
                sns.scatterplot(data=df, x="pca-one", y="pca-two", hue="label")
                plt.title("PCA")
                if save == True:
                    plt.savefig("../Data/png/pca_result_2dim.png")
                plt.show()
                
            if dim == 3:
                colors = {"0":"red", "1":"green"}
                fig = plt.figure(figsize = (8,8))
                ax = fig.add_subplot(projection='3d')
                ax.scatter(df['pca-one'],
                        df['pca-two'],
                        df['pca-three'],  
                        alpha=1, 
                        label=df["label"],
                        c=df["label"].map(colors))
                ax.set_xlabel('pca1')
                ax.set_ylabel('pca2')
                ax.set_zlabel('pca3')
                if save == True:
                    plt.savefig("../Data/png/pca_result_3dim.png")
                plt.show()
                

        except Exception as e:
            print("Exception raised : ", e)


    def Kmeans(self, df, n_cluster=2, show_plot=False, cm= False, save=False):
        self.kmeans = KMeans(init="k-means++", n_clusters=n_cluster, random_state=42)
        y_pred = self.kmeans.fit_predict(df[["pca-one","pca-two"]])

        df["cluster"] = y_pred

        acc_score = accuracy_score(y_true=df["label"].astype("int"), y_pred=y_pred)
        print(f"Kmeans 클러스터링 결과와 실제 label은 {round(acc_score,3)* 100}% 만큼 일치합니다")
        print("===" * 30)
        print(classification_report(df["label"].astype("int"), y_pred))

        if show_plot == True:
                
            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(111, projection='3d')

            ax.scatter(df.loc[:,"pca-one"]
                    , df.loc[:,"pca-two"]
                    , df.loc[:,"pca-three"]
                    , c = df['cluster']
                    , s = 20
                    , cmap = "rainbow"
                    , alpha = 1
                    )
            plt.show()
            if save == True:
                plt.savefig("../Data/png/kmeans_plot_3d.png")
                        
        if cm == True:
            cm = confusion_matrix(df["label"].astype("int"), y_pred)
            sns.heatmap(cm, annot=True) 
            plt.show()

            if save == True:
                plt.savefig("../Data/png/confusion_mtx.png")


        return df
    





            