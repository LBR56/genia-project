from konlpy.tag import Mecab
from tqdm.notebook import tqdm
import kss
from hanspell import spell_checker
import re
import pandas as pd
from Preprocess.load import DataLoad


def get_list_dataset(popular_clean_df, unpopular_clean_df):
    """
    Func description 
        : dataframe 형태의 데이터를 받아 리스트 형태의 corpus를 반환
    Params
        popular_clean_df : clean dataframe of popular lecture
        unpopular_clean_df : clean dataframe of unpopular lecture
    Returns
        popular_list : list data of popular lecture
        unpopular_list : list data of unpopular lecture
        texts : list of combined corpus from popular & unpopular lecture
    """
    popular_list = [doc.replace("/", " ") for doc in popular_clean_df["text"]]
    unpopular_list = [doc.replace("/", " ") for doc in unpopular_clean_df["text"]]
    texts = [" ".join(sent) for sent in (doc.split('/') for doc in popular_clean_df["text"])] + [" ".join(a) for a in (i.split('/') for i in unpopular_clean_df["text"])]

    return popular_list, unpopular_list, texts


def tokenizer(sentence, mode="noun", pos=['NP','NNG', 'NNP']): 
    """
    Func description
        : sentence를 받아 형태소 분석을 실시합니다
    Params
        sentence : target sentence for tokenizing
        mode : "noun" or "morphs". Default: noun
        pos : get only tag for NP,NNG,NNP
    Return:
        tokens : list of tokens
    """

    stopwords = DataLoad.load_stopwords()

    if mode == "noun":
        tokenized = Mecab().pos(sentence)
        tokens = [word for word, tag in tokenized if tag in pos and len(word) > 1]
        tokens = [word for word in tokens if word not in stopwords]
        tokens = list(map(Preprocess.regex_preprocess, tokens))
        return tokens
    
    if mode == "morphs":
        tokenized = Mecab().morphs(sentence)
        tokens = [word for word in tokenized if len(word) > 1]
        tokens = [word for word in tokens if word not in stopwords]
        tokens = list(map(Preprocess.regex_preprocess, tokens))
        return tokens


class Preprocess:
    """
    Class description : dataframe 형태로 저장된 데이터를 불러와 모델에 맞게 전처리를 진행합니다
    Params:
        stopwords : list of stopwords
    """
    def __init__(self, stopwords):
        self.stopwords = stopwords

    @staticmethod
    def regex_preprocess(text):
        """
        Func description : 한국어 이외의 데이터를 제외합니다
        """
        pattern = '[^ ㄱ-ㅣ가-힣]+'
        text = re.sub(pattern,'', text)
        text = re.sub("  ", " ", text)
        return text
    
    @staticmethod
    def get_cleaned_df(dataframe, save=False, file_nm=None):
        """
        Func Description : Raw 형태의 dataframe 데이터에 대해 문장 분리 및 맞춤법 검사를 실시합니다
        param
            dictionary : target dataframe for clean
        return
            dictionary : dataframe cleaned
        """
        for idx in dataframe.index:
            text_checked = list()
            texts = dataframe.loc[idx, "text"]
            sentence_split = kss.split_sentences(texts)
            sentences = list(map(Preprocess.regex_preprocess, sentence_split))
            for sen in tqdm(sentences):
                spelled_sent = spell_checker.check(sen).checked
                text_checked.append(spelled_sent)
            dataframe.loc[idx, "text"] = "/".join(text_checked)

            if save==True:
                dataframe.to_csv(f"../Data/csv/{file_nm}.csv")

        return dataframe

        



    
    
    









