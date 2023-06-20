import pandas as pd
import os
from Preprocess.load import DataLoad
from Preprocess.preprocess import Preprocess
from Preprocess import preprocess
from Model.model import TfIdf, Doc2Vec, PcaCluster
import warnings
warnings.filterwarnings(action="ignore")

def main():
    loader = DataLoad(
    tran_path="/Users/jylee/Desktop/GeniA_project/LJY/Data/csv/sample_transcripts.csv",
    meta_path="/Users/jylee/Desktop/GeniA_project/LJY/Data/csv/sample_videos.csv"
    )

    texts_df_popular, texts_df_unpopular = loader.get_data_set(save=False)
    stopwords = loader.load_stopwords()
    math_list = loader.load_math_list()
    preprocessor = Preprocess(stopwords)
    # popular_clean_df = preprocessor.get_cleaned_df(texts_df_popular, save=True, file_nm="popular_clean_df")
    # unpopular_clean_df = preprocessor.get_cleaned_df(texts_df_unpopular, save=True, file_nm="unpopular_clean_df")

    popular_clean_df = pd.read_csv("./LJY/Data/csv/popular_clean_df.csv", index_col=0)
    unpopular_clean_df = pd.read_csv("./LJY/Data/csv/unpopular_clean_df.csv", index_col=0)
    print("cleaned dataframes are ready!")
    popular_list, unpopular_list, texts = preprocess.get_list_dataset(popular_clean_df, unpopular_clean_df)



    tfidf = TfIdf(
    texts=texts,
    tokenizer=preprocess.tokenizer
    )
    popular_tfidf, unpopular_tfidf = tfidf.transform_into_vector(popular_list, unpopular_list)
    po_kw_df, unpo_kw_df = tfidf.get_keywords(popular_tfidf=popular_tfidf, unpopular_tfidf=unpopular_tfidf, top_n=10)
    print("TF-IDF keywords are ready!")

    cnt_po, cnt_unpo = 0,0
    for i in range(len(po_kw_df)):
        count_po = po_kw_df.loc[i].apply(lambda x: x not in math_list).sum()
        count_unpo = unpo_kw_df.loc[i].apply(lambda x: x not in math_list).sum()
        cnt_po += count_po
        cnt_unpo += count_unpo


    print("TF-IDF의 중요단어 중 수학용어가 나오지 않은 수")
    print("="*50)
    print(f"인기강좌 : {cnt_po}")
    print(f"비인기강좌 : {cnt_unpo}")
    
    print("="* 50)
    print("Doc2Vec analysis starts")

    doc2vec = Doc2Vec(
    popular_list=popular_list,
    unpopular_list=unpopular_list
    )
    total_doc_df, tagged_corpus_list = doc2vec.get_doc2vec_sets()
    doc2vec.fit(tagged_corpus_list=tagged_corpus_list)
    doc2vec.get_similarity_rate(save=False)

    print("PCA analysis starts")
    pca_cluster = PcaCluster(pca_n_components=3, random_state=42)
    df_pca = pca_cluster.get_df_pca(doc2vec.model.dv.vectors)
    df = pca_cluster.fit_transform(df_pca)
    print("PCA dataframe is ready")
    pca_cluster._plot_pca_results(df, dim=3, save=False)
    print("PCA plot has been saved successfully!")
    df_kmeans = pca_cluster.Kmeans(df, show_plot=True, cm=True, save=False)
    print("PCA plot 2 and confusion matrix have been saved successfully!")


if __name__ == "__main__":
    main()