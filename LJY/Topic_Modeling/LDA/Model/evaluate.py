import numpy as np
import pandas as pd
import pyLDAvis.gensim_models as gensimvis
import pyLDAvis



def get_best_topics_df(corpus, texts, model_list, coherence_values, perplexity_values, mode="coherence"):

    """
    Func description : 해당하는 지표를 기준으로 한 LDA 모델의 토픽추출 결과를 DataFrame으로 나타냅니다

    Params 
        corpus : Gensim corpus
        texts : original texts
        model_list : LDA model lists from experiments
        coherence_values : list of coherence values from each model in list
        perplexity_values : list of perplexity values from each model in list

    Returns
        sent_topics_df : 각 corpus의 가장 높은 확률의 토픽 , 확률, 토픽 keywords, Original texts를 담고 있는 Dataframe
    """

    best_model_by_coherence = model_list[np.argmax(coherence_values)]
    best_model_by_perplexity = model_list[np.argmin(perplexity_values)]

    if mode == "coherence":
        ldamodel = best_model_by_coherence
    
    if mode == "perplexity":
        ldamodel = best_model_by_perplexity
        
    sent_topics_df = pd.DataFrame(columns=['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords'])

    for idx, row in enumerate(ldamodel[corpus]):
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        for j, (topic_number, prob_topic) in enumerate(row):
            if j == 0:
                topics = ldamodel.show_topic(topic_number)
                topic_keywords = ", ".join([word for word, prob in topics])
                append_list = [int(topic_number), round(prob_topic, 4), topic_keywords]
                sent_topics_df.loc[idx] = append_list
            else:
                break

    sent_topics_df["Original_Texts"] = texts

    return sent_topics_df


def get_topics_df(corpus, texts, ldamodel):
    

    sent_topics_df = pd.DataFrame(columns=['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords'])

    for idx, row in enumerate(ldamodel[corpus]):
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        for j, (topic_number, prob_topic) in enumerate(row):
            if j == 0:
                topics = ldamodel.show_topic(topic_number)
                topic_keywords = ", ".join([word for word, prob in topics])
                append_list = [int(topic_number), round(prob_topic, 4), topic_keywords]
                sent_topics_df.loc[idx] = append_list
            else:
                break

    sent_topics_df["Original_Texts"] = texts

    return sent_topics_df
    



def lda_visualize(model, corpus, dictionary, file_nm):
    visualization = gensimvis.prepare(model, corpus, dictionary, sort_topics=False)
    pyLDAvis.save_html(visualization, f"./Data/html/{file_nm}.html")




        
