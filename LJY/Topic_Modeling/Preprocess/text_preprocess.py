from Preprocess.load import get_cleaned, load_data
from konlpy.tag import Mecab
from tqdm.notebook import tqdm
from gensim import corpora



class Preprocess:
    """
    Class description : dataframe으로 부터 추출된 데이터를 LDA 모델의 Input 데이터로 전처리
    Params:
        file_nm : file_name with path
        tokenize_type : Set type of tokenizing (default : nouns / option : nouns, morphs)
    """
    def __init__(self, file_nm, tokenize_type="nouns"):
        self.tokenizer = Mecab()
        self.file_nm = file_nm
        self.dataframe = load_data(self.file_nm)
        self.sentences = get_cleaned(self.dataframe)
        self.tokenize_type = tokenize_type



    @staticmethod
    def get_nouns(tokenizer, sentence):
        """
        Func Description : Tokenizer를 사용해 여러 종류의 필요한 명사만을 추출합니다
        param
            tokenizer : 사용 할 형태소 분석기
            sentence : tokenize할 문장
        return
            nouns : 길이가 2이상인 명사를 포함하는 List
        """
        tokenized = tokenizer.pos(sentence)
        nouns = [[word for word, tag in tokenized if tag in ['NP', 'SN', 'SL', 'NNG', 'NNP'] and len(word) > 1]]
        return nouns

    @staticmethod
    def get_morphs(tokenizer, sentence):
        """
        Func Description : Tokenizer를 사용해 여러 종류의 필요한 명사만을 추출합니다
        param
            tokenizer : 사용 할 형태소 분석기
            sentence : tokenize할 문장
        return
            nouns : 길이가 2이상인 명사를 포함하는 List
        """
        tokenized = tokenizer.morphs(sentence)
        tokenized = [[word for word in tokenized if len(word) > 1]]

        return tokenized


    def tokenize(self):
        """
        Func Description : tokenize를 실시합니다

        return
            processed_data : 토큰화가 완료된 corpus
        """
        tokenizer = self.tokenizer
        processed_data = list()

        # for sentence in tqdm(self.sentences):
        #     if self.tokenize_type == "nouns":
        #         nouns = Preprocess.get_nouns(tokenizer, sentence)
        #         if not nouns:
        #             continue
        #         processed_data.append(nouns)
        #     if self.tokenize_type == "morphs":
        #         morphs = Preprocess.get_morphs(tokenizer, sentence)
        #         if not morphs:
        #             continue
        #         processed_data.append(morphs)
        if self.tokenize_type == "nouns":
            sentences = Preprocess.get_nouns(tokenizer, self.sentences)

        if self.tokenize_type == "morphs":
            sentences = Preprocess.get_morphs(tokenizer, self.sentences)

        # return processed_data
        return sentences



    def get_lda_inputs(self):
        """
        Func descrption : LDA 모델에 필요한 input set 준비
        
        Returns
            texts : 전처리 된 texts
            dictionary : Gensim dictionary
            corpus : Gensim corpus
        """
        texts = self.tokenize()
        dictionary = corpora.Dictionary(texts)
        # dictionary.filter_extremes(no_below=5)
        corpus = [dictionary.doc2bow(text) for text in texts]

        return texts, dictionary, corpus
