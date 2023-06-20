import pandas as pd
import csv
import os



class DataLoad:
    """
    Class description 
        : csv 형태인 데이터들을 불러와 원하는 데이터를 불러오며, 분석 목적에 맞는 형태의 dictionary 데이터를 생성합니다

    Params
        init : 
            tran_path : path of the transcript data
            meta_path : path of the video meta data
    """
    def __init__(self, tran_path:str, meta_path:str):
        self.tran_path = tran_path
        self.meta_path = meta_path


    def load_transcripts(self): 
        """
        Func description : Load transcripts
        Returns
            dataframe : dataframe of transcript data
        """
        dataframe = pd.read_csv(self.tran_path, index_col=0)
        return dataframe

    def load_meta(self):
        """
        Func description : Load video meta
        Returns
            view_info : dataframe of meta data, sorted by view counts
        """        
        dataframe = pd.read_csv(self.meta_path, index_col=0)
        view_info = dataframe[["id","viewCount"]].sort_values(by="viewCount", ascending=False).reset_index()
        return view_info


    
    def get_data_set(self, save=False):
        """
        Func description : creating Dataframe format data of texts from popular & unpopular videos
        Returns
            texts_dict_popular : dataframe of popular video texts data
            texts_dict_unpopular : dataframe of unpopular video texts data
        """    

        transcripts = self.load_transcripts()
        meta = self.load_meta()
        idx = [i for i in meta['id'] if i not in transcripts["video_id"].tolist()]
        meta = meta[meta["id"].apply(lambda x : x not in idx)]


        popular = meta[:30]
        unpopular = meta[-30:]


        texts_df_popular = pd.DataFrame(columns=["video_id", "text"])
        texts_df_unpopular = pd.DataFrame(columns=["video_id", "text"])


        for idx, _id in enumerate(popular["id"]):
            texts = transcripts[transcripts["video_id"] == _id]["text"].tolist()
            texts = "".join(texts)
            texts_df_popular.loc[idx, "video_id"] = _id
            texts_df_popular.loc[idx, "text"] = texts

        for idx, _id in enumerate(unpopular["id"]):
            texts = transcripts[transcripts["video_id"] == _id]["text"].tolist()
            texts = "".join(texts)
            texts_df_unpopular.loc[idx, "video_id"] = _id
            texts_df_unpopular.loc[idx, "text"] = texts

        if save == True:
            texts_df_popular.to_csv("/LJY/Data/csv/texts_popular.csv")
            texts_df_unpopular.to_csv("/LJY/Data/csv/texts_unpopular.csv")

        return texts_df_popular, texts_df_unpopular

    @staticmethod
    def load_stopwords():
        """
        Func description : get stopwords
        Returns
            stopwords : return list of stopwords
        """
        stopwords = pd.read_excel("./LJY/Data/stop_words.xlsx")
        stopwords = stopwords['stopword'].tolist()
        return stopwords
    
    @staticmethod
    def load_math_list():
        math_list = pd.read_csv("./LJY/Data/csv/math_dict.csv")["dict"].tolist()
        return math_list
        




