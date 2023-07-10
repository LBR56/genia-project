import pandas as pd
import math
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
        Return
            transcripts : dataframe of transcript data
        """
        transcripts = pd.read_csv(self.tran_path, index_col=0)
        return transcripts

    def load_meta(self):
        """
        Func description : Load video meta
        Return
            meta_data : dataframe of meta data, sorted by view counts
        """        
        dataframe = pd.read_csv(self.meta_path, index_col=0)
        meta_data = dataframe.sort_values(by="viewCount", ascending=False).reset_index()
        return meta_data

    @staticmethod
    def get_id(meta:pd.DataFrame,ratio=0.3):
        """
        Func description : 인기 / 비인기 강좌의 video_id를 획득
        Param
            meta : 유튜브 영상의 meta data
            ratio : 전채 중 인기 / 비인기의 차지 비율 
        Return
            popular_id : 인기 강의들의 id
            nonpopular_id : 비인기 강의들의 id 
        """
        split_num = math.floor(meta.shape[0]*ratio)
        popular_id = meta[:split_num]['id'].to_list()
        unpopular_id = meta[-split_num:]['id'].to_list()

        return popular_id, unpopular_id

    @staticmethod
    def get_df(transcript:pd.DataFrame, movie_id:list) :
        """
        Func description : 특정 id의 transcript data를 획득
        Param 
            transcript : 수집한 전체 transcript 데이터
            movie_id : 획득할 id 리스트
        Return
            transcript_df : 인기 강좌들의 transcript
        """
        transcript_df = transcript[transcript['video_id'].isin(movie_id)]
        
        return transcript_df

    @staticmethod
    def get_script(transcript:pd.DataFrame) : 
        """
        Func description : DataFrame에서 id별 text 추출
        Param
            transcript : transcript.csv 파일 
        Return
            script : video_id 별 script dictionary
        """
        script = transcript.groupby('video_id').sum().to_dict()['text']     
        
        return script