from Model.model import NER, Formal
import pandas as pd 
from collections import Counter


def Formalprocess(script) : 
    model = Formal()
    script_list = script.split('/')
    formal_check = 0 
    for sentence in script_list :
        if model.pred(sentence) == 1 : 
            formal_check += 1 
    formal_ratio = formal_check / len(script_list)
    return formal_ratio

def NERprocess(ckpt_path,conf_path,script) : 
    model = NER(ckpt_path,conf_path)
    script_list = script.split('/')
    label_list = [] 
    for sentence in script_list :
        label_list += model.inference_fn(sentence)
    return label_list

def exist_label(label_list) : 
    new_label_list = []
    for label in label_list :
        if all([label != 'O',  label[-2:] != '-I']): 
            new_label_list.append(label[:3])
    return new_label_list

def NER_ratio(df) : 
    all_label = df['exist_label'].sum()
    all_label = dict(Counter(all_label))
    all_label = dict(sorted(all_label.items()))
    label_total = sum(all_label.values())
    ratio_data = {key: value / label_total for key, value in all_label.items()} 

    return ratio_data

