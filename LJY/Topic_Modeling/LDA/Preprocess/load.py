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

    def get_data_set(self):
        """
        Func description : creating dictionary format data of texts from popular & unpopular videos
        Returns
            texts_dict_popular : dictionary of popular video texts data
            texts_dict_unpopular : dictionary of unpopular video texts data
        """    
        popular = self.load_meta()[:30]
        unpopular = self.load_meta()[-30:]
        transcripts = self.load_transcripts()
        texts_dict_popular = dict()
        texts_dict_unpopular = dict()


        for idx, _id in enumerate(popular["id"]):
            texts = transcripts[transcripts["video_id"] == _id]["text"].tolist()
            texts_dict_popular[str(idx)] = texts

        for idx, _id in enumerate(unpopular["id"]):
            texts = transcripts[transcripts["video_id"] == _id]["text"].tolist()
            texts_dict_unpopular[str(idx)] = texts

        return texts_dict_popular, texts_dict_unpopular

    @staticmethod
    def load_stopwords():
        """
        Func description : get stopwords
        Returns
            stopwords : return list of stopwords
        """
        stopwords = pd.read_excel("/Users/jylee/Desktop/GeniA_project/Topic_Modeling/LDA/Data/stop_words.xlsx")
        stopwords = stopwords['stopword'].tolist()
        return stopwords




def save_processed_data(processed_data, file_nm):
    """
    Func Description : 토큰 분리한 데이터를 csv로 저장
    Param 
        processed_data: 전처리 된 texts
    return:
    """
    path = os.path.dirname(r"./Data/csv/")
    with open(path + '/' + file_nm, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        for data in processed_data:
            writer.writerow(data)

def save_corpus(corpus, file_nm):
    """
    Func Description : 토큰 분리한 데이터를 csv로 저장
    Param 
        processed_data: 전처리 된 texts
    return:
    """
    path = os.path.dirname(r"./Data/csv/")
    with open(path + '/' + file_nm, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        for data in corpus:
            writer.writerow(data)

def save_dataframe(dataframe, file_nm):
    dataframe.to_csv("./Data/csv/" + file_nm)
    
