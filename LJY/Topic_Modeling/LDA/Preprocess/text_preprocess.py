from konlpy.tag import Mecab
from tqdm.notebook import tqdm
from gensim import corpora
import kss
from hanspell import spell_checker
import re
import itertools


class Preprocess:
    """
    Class description : Dictionary 형태로 저장된 데이터를 불러와 모델에 맞게 전처리를 진행합니다
    Params:
        stopwords : list of stopwords
        tokenize_type : Set type of tokenizing (default : nouns / option : nouns, morphs)
    """
    def __init__(self, stopwords, tokenize_type="nouns"):
        self.stopwords = stopwords
        self.tokenizer = Mecab()
        self.tokenize_type = tokenize_type

    @staticmethod
    def regex_preprocess(text):
        """
        Func description : 한국어 이외의 데이터를 제외합니다
        """
        pattern = '[^ ㄱ-ㅣ가-힣]+'
        text = re.sub(pattern,'', text)
        return text
    
    @staticmethod
    def get_cleaned_dict(dictionary):
        """
        Func Description : Raw 형태의 dictionary 데이터에 대해 문장 분리 및 맞춤법 검사를 실시합니다
        param
            dictionary : target dictionary for clean
        return
            dictionary : dictionary cleaned
        """
        for i in range(len(dictionary)):
            text_checked = list()
            texts = "".join(dictionary[str(i)])
            sentence_split = kss.split_sentences(texts)
            sentences = list(map(Preprocess.regex_preprocess, sentence_split))
            for sen in tqdm(sentences):
                spelled_sent = spell_checker.check(sen).checked
                text_checked.append(spelled_sent)
            dictionary[str(i)] = text_checked

        return dictionary


    @staticmethod
    def get_nouns(tokenizer, stopwords, sentence):
        """
        Func Description : Tokenizer를 사용해 여러 종류의 필요한 명사만을 추출합니다
        param
            tokenizer : function used for tokenizing
            stopwords : list of stopwords
            sentence : sentence to be tokenized
        return
            tokens : list of tokens
        """
        tokenized = tokenizer.pos(sentence)
        tokens = [word for word, tag in tokenized if tag in ['NP','NNG', 'NNP'] and len(word) > 1]
        tokens = [word for word in tokens if word not in stopwords]
        return tokens


    @staticmethod
    def get_morphs(tokenizer, stopwords, sentence):
        """
        Func Description : Tokenizer를 사용해 형태소 분리를 실시합니다
        param
            tokenizer : function used for tokenizing
            stopwords : list of stopwords
            sentence : sentence to be tokenized
        return
            tokens : list of tokens
        """
        tokenized = tokenizer.morphs(sentence)
        tokens = [word for word in tokenized if len(word) > 1]
        tokens = [word for word in tokens if word not in stopwords]
        
        return tokens


    def tokenize(self, dictionary):
        """
        Func Description : tokenize를 실시합니다
        Params
            dictionary : dictionary to be tokenized
        return
            dictionary : dictionary with tokenized data
        """
        tokenizer = self.tokenizer
        stopwords = self.stopwords
        for i in range(len(dictionary)):
            token_lst = list()
            for sentence in tqdm(dictionary[str(i)]):

                if self.tokenize_type == "nouns":
                    tokens = Preprocess.get_nouns(tokenizer=tokenizer, 
                                                stopwords=stopwords, 
                                                sentence=sentence)
                if self.tokenize_type == "morphs":
                    tokens = Preprocess.get_morphs(tokenizer=tokenizer, 
                                                stopwords=stopwords, 
                                                sentence=sentence)
                if not tokens:
                    continue
                token_lst.append(tokens)

            dictionary[str(i)] = list(itertools.chain(*token_lst))

        return dictionary



    def get_lda_inputs(self, dictionary, filter=False, no_below=5, no_above=0.7):
        """
        Func descrption : LDA 모델에 필요한 input set을 반환합니다
        
        Returns
            texts : 전처리 된 texts
            dictionary : Gensim dictionary
            corpus : Gensim corpus
        """
        texts = [doc for doc in dictionary.values()]
        dictionary = corpora.Dictionary(texts)
        corpus = [dictionary.doc2bow(text) for text in texts]
        
        if filter == True:
            dictionary.filter_extremes(no_below=no_below,
                                       no_above=no_above)

        return texts, dictionary, corpus







