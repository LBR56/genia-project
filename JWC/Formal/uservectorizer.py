import numpy as np

from konlpy.tag import Okt, Kkma 
from usertokenizer import UserTokenizers
import joblib
from tqdm import tqdm

from sklearn.feature_extraction.text import TfidfVectorizer


class TrainTransformVect :
    """
    class info : 자연어를 벡터화하는 훈련과 데이터 변환을 수행
    process :
        1) self.fit_run 수행
        2) self.transform_run 수행
    """
    
    def __init__(self) :
        self.ut_cls = UserTokenizers()
        
        return self.bpe_tfidf
    
    def fitKKMA_TFIDF(self) :
        """
        func info : konlpy.Kkma 토크나이저로 tfidf 벡터화
        주의 : 메모리 에러, 소요시간 주의 
        """
        self.kkma_tfidf = TfidfVectorizer(tokenizer=self.ut_cls.konlpyNounsTokenizer)
        
        return self.kkma_tfidf
    
    def fitMP_TFIDF(self) :
        """
        func info : konlpy.Okt 토크나이저로 tfidf 벡터화
        주의 : 메모리 에러, 소요시간 주의 
        """
        self.mp_tfidf = TfidfVectorizer(tokenizer=self.ut_cls.konlpyMorphsTokenizer)
        
        return self.mp_tfidf
        
    def fit_run(self, user_token_nm, data) :
        """
        func info : tfidf 벡터화 수행
        param user_token_nm : 토크나이저 선택
                            {'kkma':Kkma, 'mp':Okt}
        param data : 벡터로 변환하려는 데이터 
        """
        if user_token_nm == 'kkma' :
            self.vec_model = self.fitKKMA_TFIDF()
        elif user_token_nm == 'mp' :
            self.vec_model = self.fitMP_TFIDF()
        else :
            raise ValueError("user_token_nm이 올바르지 않습니다.['kkma','mp']중 하나가 맞는지 확인하세요.")
            self.vec_model = self.fitWP_TFIDF() # 기본 값
            
        self.vec_model.fit(data)
        
    
    def transform_run(self, data, chunk_size) :
        """
        info : tfidf 벡터화하여 np.array로 변환
        param data : 벡터로 변환하려는 데이터
        chunk_size : np.array로 변환하는 단위, 데이터 수 
        return vec_arr : np.array로 변환된 데이터 
        """
        # 데이터 수를 조정하여 데이터 변환함
        data_len = len(data)
        
        for st_idx in tqdm(range(0,data_len,chunk_size)) :
            tmp_data = data[st_idx:st_idx+chunk_size]
            
            if st_idx == 0 :
                vec_arr = self.vec_model.transform(tmp_data).toarray()
            else :
                tmp_data_arr = self.vec_model.transform(tmp_data).toarray()
                vec_arr = np.append(vec_arr, tmp_data_arr, axis = 0)
                
        return vec_arr 