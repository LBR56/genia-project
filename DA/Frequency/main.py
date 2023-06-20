import controller
import warnings
warnings.filterwarnings(action="ignore")

def main():
    total_po_nouns, total_unpo_nouns = controller.get_nouns()
    total_new_po_nouns, total_new_unpo_nouns = controller.get_nouns(without_math=True)
    
    controller.plot_most_commons(
        po_nouns=total_new_po_nouns,
        unpo_nouns=total_new_unpo_nouns,
        without_math=True,
        save=True
    )

    controller.plot_wordcloud(
        po_nouns=total_new_po_nouns,
        unpo_nouns=total_new_unpo_nouns,
        without_math=True
    )

    words_df, unique_list = controller.get_frequency_df(
        total_new_po_nouns, 
        total_new_unpo_nouns
        )

    # models = controller.Models(
    #     words_df=words_df,
    #     new_po_nouns=total_new_po_nouns,
    #     new_unpo_nouns=total_new_unpo_nouns
    # )

    # models.ols()
    # models.linear_regression()
    # models.xgboost()

    # models.xgboost_plot(unique_list, save=False)

    # models.important_words()

    tfidf_models = controller.TfidfModels(
        new_po_nouns=total_new_po_nouns,
        new_unpo_nouns=total_new_unpo_nouns,
    )
    tfidf_models.ols()
    tfidf_models.linear_regression()
    tfidf_models.xgboost()

    tfidf_models.xgboost_plot()

    tfidf_models.important_words()

if __name__ == "__main__":
    print("Frequency Report Starts")
    main()
    print("Frequency Reports Ends")

