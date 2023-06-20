import re

from konlpy.tag import Mecab
import kss
from tqdm import tqdm
import pandas as pd
from hanspell import spell_checker

from aws import AwsController

tqdm.pandas()

class Preprocesser():
    """
    Class description : Dictionary 형태로 저장된 데이터를 불러와 모델에 맞게 전처리를 진행합니다
    Params:
        stopwords : list of stopwords
        tokenize_type : Set type of tokenizing (default : nouns / option : nouns, morphs)
    """

    def __init__(self):
        aws_controller = AwsController()
        temp_df = aws_controller.s3_download(["stop_words.csv", "math_dict.csv"])
        self.stopwords = temp_df["stop_words"]["stopword"].tolist()
        self.math_list = temp_df["math_dict"]["dict"].tolist()
        self.tokenizer = Mecab()

    @staticmethod
    def regex_preprocess(text):
        """
        Func description : 한국어 이외의 데이터를 제외합니다
        """
        text = re.sub(r"\[음악\]", r"", text)
        text = re.sub(r"[^ ㄱ-ㅣ가-힣]+", r"", text)
        text = re.sub(r" +", r" ", text)
        return text

    def tokenizer(self, sentence, mode="noun", pos=['NP','NNG', 'NNP']): 
        stopwords = self.stopwords

        if mode == "noun":
            tokenized = self.tokenizer.pos(sentence)
            tokens = [word for word, tag in tokenized if tag in pos and len(word) > 1]
        elif mode == "morphs":
            tokenized = self.tokenizer.morphs(sentence)
            tokens = [word for word in tokenized if len(word) > 1]
        else:
            raise ValueError("mode must noun or morphs!")
        
        tokens = [word for word in tokens if word not in stopwords]
        tokens = list(map(Preprocesser.regex_preprocess, tokens))
        return tokens
    
    def get_preprocessed_df(self, videos, transcripts):
        temp_series = transcripts.groupby("video_id")["text"].sum()
        temp_df = pd.merge(videos, temp_series, left_on="id", right_on="video_id")

        temp_df["texts"] = temp_df["text"].apply(kss.split_sentences)
        temp_df.drop(["text"], axis=1)
        temp_df["texts"] = temp_df["texts"].apply(
            lambda x : [Preprocesser.regex_preprocess(i) for i in x]
            )
        
        temp_df["texts"] = temp_df["texts"].progress_apply(
            lambda x : [spell_checker.check(i).checked for i in x]
        )

        temp_df["texts"] = temp_df["texts"].apply(
            lambda x : "/".join(x)
        )

        return temp_df