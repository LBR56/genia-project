import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from konlpy.tag import Mecab
from Preprocess import load
from Preprocess.load import DataLoad
from Preprocess.preprocess import Preprocess
from Preprocess import preprocess
from Model.model import TfIdf

from collections import Counter
from wordcloud import WordCloud 

from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from xgboost import XGBClassifier

from matplotlib import rc  
rc('font', family='AppleGothic') 


def top_n_common_dict(top_n, counter):
    return dict(counter.most_common(top_n))

def get_nouns(without_math=False):
    loader = DataLoad(
    tran_path="./LJY/Data/csv/sample_transcripts.csv",
    meta_path="./LJY/Data/csv/sample_videos.csv"
    )
    texts_df_popular, texts_df_unpopular = loader.get_data_set(save=False)

    math_list = loader.load_math_list()
    stopwords = loader.load_stopwords()
    preprocessor = Preprocess(stopwords)
    # popular_clean_df = preprocessor.get_cleaned_df(texts_df_popular, save=True, file_nm="popular_clean_df")
    # unpopular_clean_df = preprocessor.get_cleaned_df(texts_df_unpopular, save=True, file_nm="unpopular_clean_df")

    popular_clean_df = pd.read_csv("./LJY/Data/csv/popular_clean_df.csv", index_col=0)
    unpopular_clean_df = pd.read_csv("./LJY/Data/csv/unpopular_clean_df.csv", index_col=0)

    df_po_tokenized = preprocessor.tokenize(popular_clean_df)
    df_unpo_tokenized = preprocessor.tokenize(unpopular_clean_df)

    total_po_nouns = list(itertools.chain(*[doc for doc in df_po_tokenized["text"]]))
    total_unpo_nouns = list(itertools.chain(*[doc for doc in df_unpo_tokenized["text"]]))

    if without_math == True: 
        popular_nouns = [doc for doc in df_po_tokenized["text"]]
        unpopular_nouns = [doc for doc in df_unpo_tokenized["text"]]

        new_po_nouns = [["수학" if noun in math_list else noun for noun in nouns] for nouns in popular_nouns]
        new_unpo_nouns = [["수학" if noun in math_list else noun for noun in nouns] for nouns in unpopular_nouns]
        # total_new_po_nouns = list(itertools.chain(*new_po_nouns))
        # total_new_unpo_nouns = list(itertools.chain(*new_unpo_nouns))
        return popular_nouns, unpopular_nouns, new_po_nouns, new_unpo_nouns

    return total_po_nouns, total_unpo_nouns



def plot_most_commons(po_nouns, unpo_nouns, without_math=False, save=False):
    total_new_po_nouns = list(itertools.chain(*po_nouns))
    total_new_unpo_nouns = list(itertools.chain(*unpo_nouns))
    po_nouns_counter = Counter(total_new_po_nouns)
    unpo_nouns_counter = Counter(total_new_unpo_nouns)

    if without_math == True:
        top_30_po_common = dict(po_nouns_counter.most_common(31)[1:])
        top_30_unpo_common = dict(unpo_nouns_counter.most_common(31)[1:])
    else: 
        top_30_po_common = top_n_common_dict(30, po_nouns_counter)
        top_30_unpo_common = top_n_common_dict(30, unpo_nouns_counter)

    y_pos = np.arange(len(top_30_po_common)) 
    colors = sns.color_palette('RdPu',len(top_30_po_common))[::-1]
    plt.figure(figsize=(12,8))

    plt.subplot(1,2,1)
    plt.barh(y_pos, top_30_po_common.values(), color=colors) 
    plt.title('Popular lectures')
    plt.yticks(y_pos, top_30_po_common.keys())


    plt.subplot(1,2,2)
    plt.barh(y_pos, top_30_unpo_common.values(), color=colors) 
    plt.title('Unpopular lectures')
    plt.yticks(y_pos, top_30_unpo_common.keys())

    plt.tight_layout()
    if save == True and without_math==False:
        plt.savefig("./LJY/Data/png/most_common_30_with_math.png")
    if save==True and without_math==True:
        plt.savefig("./LJY/Data/png/most_common_30_wo_math.png")

    plt.show()

def plot_wordcloud(po_nouns, unpo_nouns, without_math=False, save=False):

    def color_func(word, font_size, position,orientation,random_state=None, **kwargs):
        return("hsl({:d},{:d}%, {:d}%)".format(np.random.randint(212,313),np.random.randint(26,32),np.random.randint(45,80)))
    
    total_new_po_nouns = list(itertools.chain(*po_nouns))
    total_new_unpo_nouns = list(itertools.chain(*unpo_nouns))
    po_nouns_counter = Counter(total_new_po_nouns)
    unpo_nouns_counter = Counter(total_new_unpo_nouns)

    if without_math == True:
        top_100_po_common = dict(po_nouns_counter.most_common(101)[1:])
        top_100_unpo_common = dict(unpo_nouns_counter.most_common(101)[1:])
    else: 
        top_100_po_common = top_n_common_dict(100, po_nouns_counter)
        top_100_unpo_common = top_n_common_dict(100, unpo_nouns_counter)


    wc1 = WordCloud(background_color = 'black', font_path='AppleGothic', color_func=color_func)
    wc1.generate_from_frequencies(top_100_po_common)

    wc2 = WordCloud(background_color = 'black', font_path='AppleGothic', color_func=color_func)
    wc2.generate_from_frequencies(top_100_unpo_common)

    plt.figure(figsize=(8,8))
    plt.subplot(2, 1, 1)
    plt.axis('off')
    plt.imshow(wc1)

    plt.subplot(2, 1, 2)
    plt.axis('off')
    plt.imshow(wc2)

    if save == True and without_math==False:
        plt.savefig("./LJY/Data/png/wordcloud_with_math.png")
    if save==True and without_math==True:
        plt.savefig("./LJY/Data/png/wordcloud_wo_math.png")

    plt.show()


def get_frequency_df(new_po_nouns, new_unpo_nouns):
    unique_list = list(set(itertools.chain(*new_po_nouns + new_unpo_nouns)))

    words_df = pd.DataFrame(columns=unique_list, index=range(60)).fillna(0)

    total_new_nouns = new_po_nouns + new_unpo_nouns
    for idx in range(len(total_new_nouns)):
        for word in total_new_nouns[idx]:
            words_df.loc[idx, word] += 1

    words_df["label"] = [1] * 30 + [0] * 30
    return words_df, unique_list




class FrequencyModels:
    def __init__(self, words_df, new_po_nouns, new_unpo_nouns):
        self.words_df = words_df
        self.X = self.words_df.drop("label", axis=1)
        self.y = self.words_df["label"]
        self.new_po_nouns = new_po_nouns
        self.new_unpo_nouns = new_unpo_nouns

    def ols(self):
        X_ = sm.add_constant(self.X)

        model = sm.OLS(self.y, X_)
        self.model_trained = model.fit()

        return self.model_trained
    
    def linear_regression(self):
        self.lin_rg = LinearRegression()
        self.lin_rg.fit(self.X, self.y)
        return self.lin_rg
    
    def xgboost(self, plot=False):
        self.model_xgb = XGBClassifier(n_estimators=500, learning_rate=0.01, max_depth=4, random_state=42)
        self.model_xgb.fit(self.X, self.y)

        
        return self.model_xgb
    
    def xgboost_plot(self, unique_list, save=False):
        df_impo = pd.DataFrame({'feature_importance': self.model_xgb.feature_importances_, 
              'feature_names': unique_list}).sort_values(by=['feature_importance'], 
                                                       ascending=False)

        # plt.figure(figsize=(8,6))
        self.xgb_top_30 = df_impo.set_index("feature_names").sort_values(by=['feature_importance'], ascending=False)[:30]
        self.xgb_top_30.plot(kind="bar",color='Green')
        plt.title('Most common important features')
        plt.xlabel('count')

        if save==True:
            plt.savefig("./LJY/Data/png/xgboost_imt.png")
        plt.show()

    
    def num_important_words(self, word_list):  
        num_mvp_words_po = len([word for sen in self.new_po_nouns for word in sen if word in word_list])
        num_mvp_words_unpo = len([word for sen in self.new_unpo_nouns for word in sen if word in word_list])

        print("인기 강좌에서 해당 단어가 나온 횟수 : {}".format(num_mvp_words_po))
        print("비인기 강좌에서 해당 단어가 나온 횟수 : {}".format(num_mvp_words_unpo))


    def important_words(self):
        df_corr_neg_30 = self.words_df.corr()["label"].sort_values(ascending=True).drop("label")[:30]
        df_corr_pos_30 = self.words_df.corr()["label"].sort_values(ascending=False).drop("label")[:30]

        lin_rg_coef = pd.Series(self.lin_rg.coef_, index=self.lin_rg.feature_names_in_)
        lin_positive_30 = lin_rg_coef.sort_values(ascending=False)[:30]
        lin_negative_30 = lin_rg_coef.sort_values(ascending=True)[:30]

        ols_positive_30 = self.model_trained.params.sort_values(ascending=False)[:30]
        ols_negative_30 = self.model_trained.params.sort_values(ascending=True)[:30]

        print("양의 상관관계와 음의 상관관계 단어들의 비교")
        self.num_important_words(df_corr_pos_30)
        print("="*50)
        self.num_important_words(df_corr_neg_30)
        print("+" *50)

        print("양의 회귀계수와 음의 회귀계수 단어들의 비교")
        self.num_important_words(lin_positive_30)
        print("="*50)
        self.num_important_words(lin_negative_30)
        print("+" *50)

        print("최소자승법 양의 계수와 음의 계수 단어들의 비교")
        self.num_important_words(ols_positive_30)
        print("="*50)
        self.num_important_words(ols_negative_30)
        print("+" *50)

        lin_ols_pos = (lin_positive_30.index).intersection((ols_positive_30.index))
        lin_ols_neg = (lin_negative_30.index).intersection((ols_negative_30.index))
        print("선형회귀 & OLS 교집합")
        self.num_important_words(lin_ols_pos)
        print("="*50)
        self.num_important_words(lin_ols_neg)
        print("+" *50)

        
        lin_ols_xgb_pos = (lin_ols_pos).intersection(self.xgb_top_30.index)
        lin_ols_xgb_neg = (lin_ols_neg).intersection(self.xgb_top_30.index)
        print("선형회귀 & OLS & XGB 교집합")
        self.num_important_words(lin_ols_xgb_pos)
        print("="*50)
        self.num_important_words(lin_ols_xgb_neg)
        print("+" *50)

        print("="*50)
        print("세가지 모델의 중요 단어 집합은 다음과 같습니다")
        print("="*50)
        print(lin_ols_xgb_pos.tolist())
        print(lin_ols_xgb_neg.tolist())

        return lin_ols_pos, lin_ols_neg, lin_ols_xgb_pos, lin_ols_xgb_neg

class TfidfModels:
    def __init__(self, new_po_nouns, new_unpo_nouns):
        self.tfidf_df = self.get_tfidf_df()
        self.X = self.tfidf_df.drop("label", axis=1)
        self.y = self.tfidf_df["label"]
        self.new_po_nouns = new_po_nouns
        self.new_unpo_nouns = new_unpo_nouns

    def get_tfidf_df(self):
        loader = DataLoad(
        tran_path="./LJY/Data/csv/sample_transcripts.csv",
        meta_path="./LJY/Data/csv/sample_videos.csv"
        )
        texts_df_popular, texts_df_unpopular = loader.get_data_set(save=False)

        # popular_clean_df = preprocessor.get_cleaned_df(texts_df_popular, save=True, file_nm="popular_clean_df")
        # unpopular_clean_df = preprocessor.get_cleaned_df(texts_df_unpopular, save=True, file_nm="unpopular_clean_df")

        popular_clean_df = pd.read_csv("./LJY/Data/csv/popular_clean_df.csv", index_col=0)
        unpopular_clean_df = pd.read_csv("./LJY/Data/csv/unpopular_clean_df.csv", index_col=0)
        popular_list, unpopular_list, texts = preprocess.get_list_dataset(popular_clean_df, unpopular_clean_df)
        self.tfidf = TfIdf(
        texts=texts,
        tokenizer=preprocess.tokenizer
        )

        popular_tfidf, unpopular_tfidf = self.tfidf.transform_into_vector(popular_list, unpopular_list)

        tfidf_df = pd.concat([pd.DataFrame(popular_tfidf), pd.DataFrame(unpopular_tfidf)])
        tfidf_df["label"] = [1] * 30 + [0] * 30

        return tfidf_df

    def ols(self):
        X_ = sm.add_constant(self.X)

        model = sm.OLS(self.y, X_)
        self.model_trained_tfidf = model.fit()

        return self.model_trained_tfidf
    
    def linear_regression(self):
        self.lin_rg = LinearRegression()
        self.lin_rg.fit(self.X, self.y)
        return self.lin_rg
    
    def xgboost(self):
        self.model_xgb = XGBClassifier(n_estimators=500, learning_rate=0.01, max_depth=4, random_state=42)
        self.model_xgb.fit(self.X, self.y)

        
        return self.model_xgb
    
    def xgboost_plot(self, save=False):

        self.word_list = self.tfidf.tfidf_vectorizer.get_feature_names_out().tolist()
        df_impo = pd.DataFrame({'feature_importance': self.model_xgb.feature_importances_, 
              'feature_names': self.word_list}).sort_values(by=['feature_importance'], 
                                                       ascending=False)

        # plt.figure(figsize=(8,6))
        self.xgb_top_30 = df_impo.set_index("feature_names").sort_values(by=['feature_importance'], ascending=False)[:30]
        self.xgb_top_30.plot(kind="bar",color='Green')
        plt.title('Most common important features')
        plt.xlabel('count')

        if save==True:
            plt.savefig("./LJY/Data/png/tfidf_xgboost_imt.png")
        plt.show()

    
    def num_important_words(self, word_list):  
        num_mvp_words_po = len([word for sen in self.new_po_nouns for word in sen if word in word_list])
        num_mvp_words_unpo = len([word for sen in self.new_unpo_nouns for word in sen if word in word_list])

        print("인기 강좌에서 해당 단어가 나온 횟수 : {}".format(num_mvp_words_po))
        print("비인기 강좌에서 해당 단어가 나온 횟수 : {}".format(num_mvp_words_unpo))


    def important_words(self):
        tfidf_lin_rg_coef = pd.Series(self.lin_rg.coef_, index=self.word_list)
        tfidf_lin_positive_30 = tfidf_lin_rg_coef.sort_values(ascending=False)[:30]
        tfidf_lin_negative_30 = tfidf_lin_rg_coef.sort_values(ascending=True)[:30]

        tfidf_ols_coef = pd.Series(self.model_trained_tfidf.params.drop("const").values, index=self.word_list)
        tfidf_ols_positive_30 = tfidf_ols_coef.sort_values(ascending=False)[:30]
        tfidf_ols_negative_30 = tfidf_ols_coef.sort_values(ascending=True)[:30]


        print("양의 회귀계수와 음의 회귀계수 단어들의 비교")
        self.num_important_words(tfidf_lin_positive_30)
        print("="*50)
        self.num_important_words(tfidf_lin_negative_30)
        print("+" *50)

        print("최소자승법 양의 계수와 음의 계수 단어들의 비교")
        self.num_important_words(tfidf_ols_positive_30)
        print("="*50)
        self.num_important_words(tfidf_ols_negative_30)
        print("+" *50)

        lin_ols_pos = (tfidf_lin_positive_30.index).intersection((tfidf_ols_positive_30.index))
        lin_ols_neg = (tfidf_lin_negative_30.index).intersection((tfidf_ols_negative_30.index))
        print("선형회귀 & OLS 교집합")
        self.num_important_words(lin_ols_pos)
        print("="*50)
        self.num_important_words(lin_ols_neg)
        print("+" *50)

        
        lin_ols_xgb_pos = (lin_ols_pos).intersection(self.xgb_top_30.index)
        lin_ols_xgb_neg = (lin_ols_neg).intersection(self.xgb_top_30.index)
        print("선형회귀 & OLS & XGB 교집합")
        self.num_important_words(lin_ols_xgb_pos)
        print("="*50)
        self.num_important_words(lin_ols_xgb_neg)
        print("+" *50)

        print("="*50)
        print("세가지 모델의 중요 단어 집합은 다음과 같습니다")
        print("="*50)
        print(lin_ols_xgb_pos.tolist())
        print(lin_ols_xgb_neg.tolist())

        return lin_ols_pos, lin_ols_neg, lin_ols_xgb_pos, lin_ols_xgb_neg


def make_freq_dict(po_nouns, unpo_nouns, word_list):
    dict_po, dict_unpo = {}, {}
    for word in word_list:
        dict_po[word] = 0
        dict_unpo[word] = 0

    for po_sen, unpo_sen in zip(po_nouns, unpo_nouns):
        for po_word, unpo_word in zip(po_sen, unpo_sen):
            if po_word in word_list: 
                dict_po[po_word] += 1

            if unpo_word in word_list:
                dict_unpo[unpo_word] += 1

    return dict_po, dict_unpo

def plot_freq_dict(po_dict, unpo_dict, save=False, file_nm=None):
    po_freq = pd.Series(po_dict).sort_values(ascending=False)[:10]
    unpo_freq = pd.Series(unpo_dict).sort_values(ascending=False)[:10]

    if len(po_dict.keys()) > 8:
        plt.figure(figsize=(10, 6))
    else:
        plt.figure(figsize=(8,6))
    bar_width = 0.25

    index = np.arange(len(po_freq))

    plt.bar(index, po_freq, bar_width, color='royalblue', label="Popular Nouns")
    plt.bar(index+bar_width, unpo_freq, bar_width, color='palevioletred', label="Unpopular Nouns")
    plt.xticks(index, po_freq.index, rotation=45)
    plt.legend()
    if save==True:
        plt.savefig(f"./LJY/Data/png/{file_nm}.png")
    
    plt.show()



def plot_relevant_nouns(po_nouns, unpo_nouns, lin_ols_pos, lin_ols_neg, lin_ols_xgb_pos, lin_ols_xgb_neg, save=False, tfidf=False):
    if tfidf == True:
        text = "tfidf_"
    else:
        text = ""

    dict_po_lin_ols_pos, dict_unpo_lin_ols_pos = make_freq_dict(po_nouns, unpo_nouns, lin_ols_pos)
    dict_po_lin_ols_neg, dict_unpo_lin_ols_neg = make_freq_dict(po_nouns, unpo_nouns, lin_ols_neg)
    dict_po_lin_ols_xgb_pos, dict_unpo_lin_ols_xgb_pos = make_freq_dict(po_nouns, unpo_nouns, lin_ols_xgb_pos)
    dict_po_lin_ols_xgb_neg, dict_unpo_lin_ols_xgb_neg = make_freq_dict(po_nouns, unpo_nouns, lin_ols_xgb_neg)


    plot_freq_dict(dict_po_lin_ols_pos, dict_unpo_lin_ols_pos, save=save, file_nm=f"{text}lin_ols_pos")
    plot_freq_dict(dict_po_lin_ols_neg, dict_unpo_lin_ols_neg, save=save, file_nm=f"{text}lin_ols_neg")
    plot_freq_dict(dict_po_lin_ols_xgb_pos, dict_unpo_lin_ols_xgb_pos, save=save, file_nm=f"{text}lin_ols_xgb_pos")
    plot_freq_dict(dict_po_lin_ols_xgb_neg, dict_unpo_lin_ols_xgb_neg, save=save, file_nm=f"{text}lin_ols_xgb_neg")