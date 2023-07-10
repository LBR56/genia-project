import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd 

def NERratio_plot(pop_ratio_data,unpop_ratio_data) : 
    fig, ax = plt.subplots(1,2, sharey=True , figsize=(12,5)) 
    sns.barplot(x = list(pop_ratio_data.keys()), y = list(pop_ratio_data.values()), ax=ax[0])
    ax[0].set_title('pop')
    sns.barplot(x = list(unpop_ratio_data.keys()), y = list(unpop_ratio_data.values()), ax=ax[1])
    ax[1].set_title('unpop')

def Formal_plot(df) : 
    fig, ax = plt.subplots(1,2, sharex=True,sharey=True , figsize=(12,5)) 
    sns.histplot(df['Formal'][:31], ax=ax[0])
    ax[0].set_title('pop')
    sns.histplot(df['Formal'][31:], ax=ax[1])
    ax[1].set_title('unpop')
    plt.title('unpopular')

def Wordbysecond_plot(df) : 
    fig, ax = plt.subplots(1,2, sharex=True,sharey=True , figsize=(12,5)) 
    sns.histplot(df['word by second'][:31], ax=ax[0])
    ax[0].set_title('pop')
    sns.histplot(df['word by second'][31:], ax=ax[1])
    ax[1].set_title('unpop')
    plt.title('unpopular')