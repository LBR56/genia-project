import pandas as pd
import kss
import re
from hanspell import spell_checker
from tqdm.notebook import tqdm
import csv
import os


def load_data(file_nm) -> str : 
    dataframe = pd.read_csv(file_nm, index_col=0)
    return dataframe


def regex_preprocess(text):
    pattern = '[^ ㄱ-ㅣ가-힣|0-9|a-zA-Z]+'
    text = re.sub(pattern,'', text)
    return text


def get_cleaned(dataframe):
    """
    Func Description : 한국어, 영어, 숫자를 제외한 불필요한 요소를 제거합니다
    param
        text : 전처리 제거의 대상 텍스트 입니다
    return
        text : 전처리가 완료된 텍스트 입니다
    """
    text = ''.join(dataframe['text'])

    # sentence tokenizing
    text = kss.split_sentences(text)
    # get clean
    sentences = list(map(regex_preprocess, text))
    
    text_checked = []
    for sen in tqdm(sentences):
        spelled_sent = spell_checker.check(sen)
        checked_sen = spelled_sent.checked
        text_checked.append(checked_sen)

    return " ".join(text_checked)

def save_processed_data(processed_data, file_nm):
    """
    Func Description : 토큰 분리한 데이터를 csv로 저장
    Param 
        processed_data: 전처리 된 texts
    return:
    """
    path = os.path.dirname(r"/Users/jylee/Desktop/GeniA_project/Topic_Modeling/Data/")
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
    path = os.path.dirname(r"/Users/jylee/Desktop/GeniA_project/Topic_Modeling/Data/")
    with open(path + '/' + file_nm, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        for data in corpus:
            writer.writerow(data)

def save_dataframe(dataframe, file_nm):
    dataframe.to_csv("/Users/jylee/Desktop/GeniA_project/Topic_Modeling/Data/" + file_nm)
    
