import kss
from hanspell import spell_checker
import re
import kss
import pandas as pd 


def regex_preprocess(text:str):
    """
    Func description : 한국어 이외의 데이터를 제외합니다
    Param 
        text : script
    Return
        text : text에서 한글 이외의 글자를 삭제 
    """
    pattern = '[^ ㄱ-ㅣ가-힣]+'
    text = re.sub(pattern,'', text)
    return text

def text_preprocess(text:str) : 
    """
    Func description : 주어진 전체 text를 정제한 후 문장단위로 분리
    Param
        text : script
    Return 
        check_sentence : 정제된 sentence list
    """

    text = text.replace('[음악]','')
    text = regex_preprocess(text)
    # text = text.replace(' ','')
    text = kss.split_sentences(text)
    for idx, sentence in enumerate(text) :
        text[idx] = spell_checker.check(sentence).checked
    
    return text

def get_processing_df(script:dict) :
    """
    Func description : script를 id_list에 매칭한 DataFrame으로 만듦
    Param 
        script : id별 script list로 묶인 dictionary 
    """

    processing_df = pd.DataFrame(columns=['id','script'])
    for id in script.keys() :
        for sentence in script[id] :
            temp_df = pd.DataFrame({'id' : id,'script':sentence},index=[0])
            processing_df = pd.concat([processing_df,temp_df]).reset_index(drop=True)
    
    return processing_df

def dutosec(duration) :
    duration = duration.split('T')[-1]
    hours, minutes, seconds = 0,0,0

    if 'H' in duration:
        hours = duration.split('H')[0]
        duration = duration.split('H')[1]
        hours = int(hours)

    if 'M' in duration:
        minutes = duration.split('M')[0]
        duration = duration.split('M')[1]
        minutes = int(minutes)

    if 'S' in duration :    
        seconds = int(duration[:-1])

    total_seconds = hours * 3600 + minutes * 60 + seconds

    return total_seconds
