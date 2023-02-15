import os
import pickle
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
import copy
import numpy as np
import pandas as pd
import sys

from transformers import AutoTokenizer, BertForSequenceClassification, BertConfig, \
    get_linear_schedule_with_warmup, BertPreTrainedModel, BertModel, BertForMaskedLM
from nltk.tokenize import sent_tokenize, word_tokenize 

from batch_st_3 import ST

bleurt_tokenizer = AutoTokenizer.from_pretrained("bleurt-large-512")
max_length = 16
f = 1
step = f 
RiskSet = {
    'punctuations': ['"', ":", "'", ",", ";"],
    'stopwords': [".", "?", "!"],
    "subwords": ['-', '_', '*', '(', ')', '{', '}', '[', ']', '《', '》', '@', '#', "$", "%", "^", "&", \
                 "~", "`", 'd', 't', 's', 'm', 'al', 'unk', '<unk>', '', "''", '``', 'but-but', 'is', 'who', 'off',
                 'entering', 'left', 'it', 'they', '80-85', 'males', 'up', 'a', 'of', 'for', 'the', 'into', 'were'],
}


def watermark_embedding(index_list, sentences_list):
    length = len(index_list)
    sen_token_list = [word_tokenize(sentence) for sentence in
                      sentences_list]  # sen_token_list= [['He', 'watches', 'his', 'favorite', 'show', 'every', 'night', 'on', 'time', '.'], ['He', 'watches', 'his', 'favorite', 'show', '.']]
    m = [] + [
        [0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1,
         1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1]] * length 

    m_to_save = [] + [[]] * length  
    FC_to_save = [] + [[]] * length  
    count = [] + [0] * length  
    index_list = index_list 
    latest_embed_index_list = [] + [0] * length
    max_length = max([len(sen) for sen in sen_token_list]) 
    print("max_length=", max_length)

    for idx, sen_token in enumerate(sen_token_list):
        sen_token_list[idx] = (idx, sen_token, len(sen_token)
   
    max_length_idx = ([len(sen) for sen in sen_token_list]).index(max([len(sen) for sen in sen_token_list]))
    
    S_w_list = copy.deepcopy(sen_token_list)  
    copy_index_list = copy.deepcopy(index_list)  
    while copy_index_list != [] + [0] * len(copy_index_list):
        rest_sentences = []
        rest_index_list = []  
        assert len(index_list) == len(S_w_list)
        for index, (idx, s_w, length) in zip(index_list, S_w_list):
            if length <= 200:
                if index <= length:
                    rest_sentences.append((idx, s_w, length))
                    rest_index_list.append((idx, index))
                else:
                    copy_index_list[idx] = 0
            else:
                if index <= 200:
                    rest_sentences.append((idx, s_w, length))
                    rest_index_list.append((idx, index))
                else:
                    copy_index_list[idx] = 0
        if copy_index_list == [] + [0] * len(copy_index_list):
            continue
        assert len(rest_sentences) == len(rest_index_list)
        target_token_list = [(idx, s_w[index - 1]) for (idxx, index), (idx, s_w, length) in
                             zip(rest_index_list, rest_sentences)]  


        flag = True 
        for (idx, target_token) in target_token_list:
            if (target_token.lower() in (RiskSet["punctuations"] + RiskSet["stopwords"] + RiskSet["subwords"])):
                index_list[idx] = index_list[idx] + 1
                flag = False
        if flag == False:
            continue

        local_context_list = [s_w[:index] for (idxx, index), (idx, s_w, length) in zip(rest_index_list,
                                                                                       rest_sentences)]  ################################################################改了
        local_index_list = [index for idx, index in rest_index_list]
        Sync_list, C_list = ST(local_index_list, local_context_list) 

        Substitutable_list = [False] * len(Sync_list)
        copy_local_context_list = copy.deepcopy(local_context_list)
        copy_local_index_list = copy.deepcopy(local_index_list)
        for idxx, (Sync, C, (idx, target_token), local_context, local_index) in enumerate(
                zip(Sync_list, C_list, target_token_list, copy_local_context_list, copy_local_index_list)):
            if (Sync == True) and (target_token in C):
                Substitutable_list[idxx] = True

        assert len(Substitutable_list) == len(rest_index_list) == len(C_list)
        for substitutable, (idx, _), C in zip(Substitutable_list, rest_index_list, C_list):
            if substitutable is True:
                m_rest = m[idx][count[idx]]
                m_to_save[idx] = m_to_save[idx] + [m_rest]  
                S_w_list[idx][1][index_list[idx] - 1] = C[m_rest]
                print("FC=", C)
                print("水印信号 m_rest=", m_rest)
                FC_to_save[idx] = FC_to_save[idx] + [(index_list[idx], C, m_rest)]
                count[idx] += 1
                latest_embed_index_list[idx] = index_list[idx]
                index_list[idx] = index_list[idx] + step
            else:
                index_list[idx] = index_list[idx] + 1
 
    for idx, S_w, length in S_w_list:
        S_w_list[idx] = ' '.join(S_w)

    return S_w_list, m_to_save, FC_to_save

def watermark_extraction(index_list, sentences_list):
    length = len(index_list)
    sen_token_list = [word_tokenize(sentence) for sentence in
                      sentences_list]  # sen_token_list= [['He', 'watches', 'his', 'favorite', 'show', 'every', 'night', 'on', 'time', '.'], ['He', 'watches', 'his', 'favorite', 'show', '.']]
    print("sen_token_list=", sen_token_list)
    m = [] + [
        [0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1,
         1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1]] * length 

    m = [] + [[]] * length  
    count = [] + [0] * length 
    FC_to_save = [] + [[]] * length 
    index_list = index_list  
    latest_embed_index_list = [] + [0] * length
    max_length = max([len(sen) for sen in sen_token_list]) 

    for idx, sen_token in enumerate(sen_token_list):
        print("sen_token=", sen_token)
        sen_token_list[idx] = (idx, sen_token, len(sen_token))

    max_length_idx = ([len(sen) for sen in sen_token_list]).index(max([len(sen) for sen in sen_token_list]))

    S_w_list = copy.deepcopy(sen_token_list) 
    copy_index_list = copy.deepcopy(index_list) 
    while copy_index_list != [] + [0] * len(copy_index_list):
        rest_sentences = []
        rest_index_list = []
        assert len(index_list) == len(S_w_list)
        for index, (idx, s_w, length) in zip(index_list, S_w_list):
            if length <= 200:
                if index <= length:
                    rest_sentences.append((idx, s_w, length))
                    rest_index_list.append((idx, index))
                else:
                    copy_index_list[idx] = 0
            else:
                if index <= 200:
                    rest_sentences.append((idx, s_w, length))
                    rest_index_list.append((idx, index))
                else:
                    copy_index_list[idx] = 0
        if copy_index_list == [] + [0] * len(copy_index_list):
            continue
        assert len(rest_sentences) == len(rest_index_list)
        target_token_list = [(idx, s_w[index - 1]) for (idxx, index), (idx, s_w, length) in
                             zip(rest_index_list, rest_sentences)]  

        flag = True 
        for (idx, target_token) in target_token_list:
            if (target_token.lower() in (RiskSet["punctuations"] + RiskSet["stopwords"] + RiskSet["subwords"])):
                index_list[idx] = index_list[idx] + 1
                flag = False
        if flag == False:
            continue

        local_context_list = [s_w[:index] for (idxx, index), (idx, s_w, length) in zip(rest_index_list,
                                                                                       rest_sentences)]  ################################################################改了
        local_index_list = [index for idx, index in rest_index_list]
        Sync_list, C_list = ST(local_index_list, local_context_list)  # Sync_list= [True, True, True]

        Substitutable_list = [False] * len(Sync_list)
        copy_local_context_list = copy.deepcopy(local_context_list)
        copy_local_index_list = copy.deepcopy(local_index_list)
        for idxx, (Sync, C, (idx, target_token), local_context, local_index) in enumerate(
                zip(Sync_list, C_list, target_token_list, copy_local_context_list, copy_local_index_list)):
            if (Sync == True) and (target_token in C):
                Substitutable_list[idxx] = True

        for substitutable, (idx, _), C in zip(Substitutable_list, rest_index_list, C_list):
            if substitutable is True:
                if sen_token_list[idx][1][index - 1] == C[0]:
                    m[idx] = m[idx] + [0]
                    FC_to_save[idx] = FC_to_save[idx] + [(index_list[idx], C, 0)]
                elif sen_token_list[idx][1][index - 1] == C[1]:
                    m[idx] = m[idx] + [1]
                    FC_to_save[idx] = FC_to_save[idx] + [(index_list[idx], C, 1)]
                latest_embed_index = index
                index_list[idx] = index_list[idx] + step
                print("m=", m)
            else:
                index_list[idx] = index_list[idx] + 1
    assert len(m) == len(FC_to_save)
    return m, FC_to_save


# 一次处理的批数据大小
length = 1

if __name__ == "__main__":
    index_list = [2, ]
    sentences_list = [['the cooperative elements incorporated into the second game were removed,']]
    sentences_list = [[' '.join(word_tokenize(sen[0]))] for sen in sentences_list]

    all_sample = []
    for sentence_list_ in sentences_list[:]:
        index_list = [2, ]
        print('index_list=', index_list, '\nsentence_list_=', sentence_list_)
        S_w_list_, m_to_save_, FC_to_save_ = watermark_embedding(index_list, sentence_list_)
        S_w_list = []
        FC_to_save = []
        m_to_save = []
        for S_w, m_to_save_item_, FC_to_save_item_ in zip(S_w_list_, m_to_save_, FC_to_save_):
            FC_to_save.append(FC_to_save_item_)
            S_w_list.append(S_w)
            m_to_save.append(m_to_save_item_)

        for s, s_w, m_to_save_item, FC_to_save_item in zip(sentence_list_, S_w_list, m_to_save, FC_to_save):
            all_sample.append((s, s_w, m_to_save_item, FC_to_save_item))
    print("*&^" * 30, "all_sample=", all_sample, "\nlength_sample=", len(all_sample))


    # 提取
    # for s_w in S_w_list[:]:
    index_list = [2, ]
    print('index_list=', index_list, '\nsentence_list_=', S_w_list)
    m, FC_to_save = watermark_extraction(index_list, S_w_list)

    print('m=', m, '\nFC_to_save=', FC_to_save)
