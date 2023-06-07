import re
from tokenizers import Tokenizer
from nltk.tokenize import RegexpTokenizer
from konlpy.tag import Okt, Kkma


class UserTokenizers :
    """ 
    calss info : 사용자의 필요에 맞게 다양한 토크나이저를 적용하는 모듈
    """
    
    def __init__(self)-> None :
        # self.bpe_tokenizer_pretrained = Tokenizer.from_file('./model/bpe_tokenizer.json')
        # self.word_tokenizer_pretrained = Tokenizer.from_file('./model/wordpiece_tokenizer.json')
        self.okt = Okt()
        self.kkma = Kkma()
    
    @staticmethod
    def whitespaceToken(data : str) -> list :
        """
        func info whitespaceToken : 공백 문자로 데이터 토큰화
        param data : 토큰화할 문자 데이터 
        return token_rs : 토큰 결과 
        """
        token_rs = data.split(' ')
        
        return token_rs
    
    @staticmethod
    def regexsplitToken(data : str, pat : str = '[\.\,!?\n]') -> list :
        """
        func info regexsplitToken : 정규표현식에 패턴을 기준으로 데이터 토큰화
        param data : 토큰화할 문자 데이터 
        param pat : 토큰화할 정규표현식
        return token_rs : 토큰 결과 
        """
        
        re_rs = re.split(pat, data, maxsplit=0)
        token_rs = [rs_unit.strip() for rs_unit in re_rs if len(rs_unit.strip()) > 1]
        
        return token_rs
    
    @staticmethod 
    def regexselectToken(data : str, pat : str = '[\w]+') -> list :
        """
        func info regexselectToken : 정규표현식의 패턴을 선택하여 데이터 토큰화
        param data : 토큰화할 문자 데이터 
        param pat : 토큰화할 정규표현식
        return token_rs : 토큰 결과 
        """
        token_rs = RegexpTokenizer(pat).tokenize(data)
        
        return token_rs
    
    def konlpyMorphsTokenizer(self, data : str) -> list :
        """
        func info konlpyMorphsTokenizer : Okt 형태소 분석 결과 
        param data : 토큰화할 문자 데이터 
        return token_rs : 토큰 결과 
        """
        
        token_rs = self.okt.morphs(data)
        
        return token_rs
    
    def konlpyNounsTokenizer(self, data : str) -> list :
        """
        func info konlpyNounsTokenizer : KKma 형태소 분석 후 명사만 추출
        param data : 토큰화할 문자 데이터 
        return token_rs : 토큰 결과 
        """
        
        token_rs = self.kkma.nouns(data)
        
        return token_rs