import os
from hanspell import spell_checker
import kss


class preprocess :
    '''
    class info : text 전처리 과정을 모아 놓은 class 
    '''

    def sentence_split(text : str) -> list :
        '''
        func info : 문장단위로 분리하는 함수

        param text : 원본 스크립트

        return sentence_list : 원본 스크립트를 전부 이어붙인후 개행문자 제거, 문장 단위로 분리한 리스트
        '''
        text = ''.join(text.split(' ')).replace('\n','')
        sentence_list = kss.split_sentences(text)
        return sentence_list
    
    def spell_check(sentence_list : list) -> list : 
        '''
        func info : list 원소마다 맞춤법 검사기를 거친 문자로 변환

        param sentence_list : 문장 단위로 분리된 list

        return fix_list : 맞춤법 검사 후 교정된 문자열 list
        '''
        fix_list = [spell_checker.check(sentence).checked for sentence in sentence_list]
        return fix_list