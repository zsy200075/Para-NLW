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
from nltk.tokenize import sent_tokenize, word_tokenize  # 分句工具

from batch_st_3 import ST

# from batch_st_3 import ST

bleurt_tokenizer = AutoTokenizer.from_pretrained("bleurt-large-512")

# ********************************************************config参数***********************************************************
max_length = 16
f = 1
step = f  # +1
RiskSet = {
    'punctuations': ['"', ":", "'", ",", ";"],
    'stopwords': [".", "?", "!"],
    "subwords": ['-', '_', '*', '(', ')', '{', '}', '[', ']', '《', '》', '@', '#', "$", "%", "^", "&", \
                 "~", "`", 'd', 't', 's', 'm', 'al', 'unk', '<unk>', '', "''", '``', 'but-but', 'is', 'who', 'off',
                 'entering', 'left' \
        , 'it', 'they', '80-85', 'males', 'up', 'a', 'of', 'for', 'the', 'into', 'were'],
    # 'is', 'it', 'on', 'of', 'date', 'a' , '<unk>', 'but' , 'he', 'she', 'it', 'we', 'him', 'his', 'her', 'hers', 'its', 'them', 'they', 'their', 'ours', 'i', 'me', 'my', 'He', 'She', 'It', 'We', 'Him', 'His', 'Her', 'Hers', 'Its', 'Them', 'They', 'Their', 'Ours', 'I', 'Me', 'My', 'There', 'there', 'Here', 'here', 'these', 'These', 'Those', 'those', 'this', 'This', 'That', 'herself', 'himself', 'themselves', 'ourselves', 'Herself', 'Himself', 'Themselves', 'Ourselves', 'you', 'your', 'You', 'Your', 'yours', 'Yours', 'for', 'to' ,'the', 'away', 'on', 'upon', 'to', 'unto', 'into', 'in', 'of', 'off', 'with', 'at' , 'toward', 'into', 'away', 'off', 'upon', 'who', '<unk>', '@-@', 'may', 'can', 'no', 'not' , 'study-', 'unk' , 'entry', 'left', ''

}


# 'date', 'on', 'at', 'to', 'of', 'that', 'the', \
#                 'with', 'of', 'a', 'an' , 'for' , 'in', 'it', 'me', 'is', 'it', 'on', 'of', '--', 'A', '...', 'gt', 'lt', \
#                 'every',
# 'the', 'with', 'of', \
#                 'a', 'an' , 'for' , 'in', 'are', 'is', 'who', 'whom', 'off', 'to', 'but-but', 'heat-', 'by', 'into'
def watermark_embedding(index_list, sentences_list):
    """
    水印嵌入过程
    参数说明：sentences_list: 一个句子列表， index_list: 一个索引列表
    返回值：S_w: 水印嵌入后的句子
    """
    length = len(index_list)
    sen_token_list = [word_tokenize(sentence) for sentence in
                      sentences_list]  # sen_token_list= [['He', 'watches', 'his', 'favorite', 'show', 'every', 'night', 'on', 'time', '.'], ['He', 'watches', 'his', 'favorite', 'show', '.']]
    #     print("sen_token_list12121212=", sen_token_list)

    # ****************************************************算法二****************************************************************
    m = [] + [
        [0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1,
         1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1]] * length  # 水印二进制位序列

    # #    指定 随机数种子，保证每次都一样
    #     np.random.seed(1)
    #     m = list(np.random.randint(0,2,(1,48)))
    #     m = [list(i) for i in m]
    #     m = m + [m[0]]*len(index_list)

    m_to_save = [] + [[]] * length  # 水印二进制位序列  = [[], [], [], [], []]
    FC_to_save = [] + [[]] * length  # = [[], [], [], [], []]
    count = [] + [0] * length  # 服务于水印二进制位序列的取值 [0, 0, 0, ]
    index_list = index_list  # 直接第二个词，没有从0索引开始
    latest_embed_index_list = [] + [0] * length
    max_length = max([len(sen) for sen in sen_token_list])  # 获取句子列表中的最长的那个句子的长度
    print("max_length=", max_length)

    for idx, sen_token in enumerate(sen_token_list):
        #         print("sen_token=", sen_token)
        sen_token_list[idx] = (idx, sen_token, len(sen_token))
    #     print("sen_token_list3333333333=", sen_token_list)   # sen_token_list3333333333= [(0, ['"', 'How', 'so', '?'], 4), (1, ['how', 'can', 'it', 'affect', 'them', '?', '"'], 7)]

    # 获取最长句子在list中的位置
    max_length_idx = ([len(sen) for sen in sen_token_list]).index(max([len(sen) for sen in sen_token_list]))
    #     print("max_length_idx=", max_length_idx)

    S_w_list = copy.deepcopy(sen_token_list)  # list
    #     print("&%^"*50, "S_w_list.type=", type(S_w_list), "S_w_list[0]=", S_w_list[0])
    copy_index_list = copy.deepcopy(index_list)  # 用来判断是否结束循环，列表元素全为0的时候结束循环

    while copy_index_list != [] + [0] * len(copy_index_list):
        # 判断index是否超出句子长度，超出的则证明已经嵌入完了，舍去
        rest_sentences = []
        rest_index_list = []  # (剩下的索引在index_list中的位置, 索引index)
        assert len(index_list) == len(S_w_list)
        for index, (idx, s_w, length) in zip(index_list, S_w_list):
            # index <= length-f说明嵌入过程还没有结束
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
            #             print("最后一步大治来啦----"*10)
            continue

        #         print("rest_sentences=", rest_sentences, "\nrest_index_list=", rest_index_list)

        assert len(rest_sentences) == len(rest_index_list)
        target_token_list = [(idx, s_w[index - 1]) for (idxx, index), (idx, s_w, length) in
                             zip(rest_index_list, rest_sentences)]  # (句子序号，目标词)

        #         print("target_token_list=", target_token_list)

        # 判断目标词中是否有无效词
        flag = True  # 不存在无效词为True
        for (idx, target_token) in target_token_list:
            #             print("target_token=", target_token, "target_next_token=", target_next_token)
            if (target_token.lower() in (RiskSet["punctuations"] + RiskSet["stopwords"] + RiskSet["subwords"])):
                index_list[idx] = index_list[idx] + 1
                flag = False
        # 如果存在无效词则重新while循环
        if flag == False:
            continue

        # 获取需要寻找替代词的句子集
        local_context_list = [s_w[:index] for (idxx, index), (idx, s_w, length) in zip(rest_index_list,
                                                                                       rest_sentences)]  ################################################################改了
        local_index_list = [index for idx, index in rest_index_list]
        #         print("*"*30, "local_index_list=", local_index_list, "\n", "*"*30, "local_context_list=", local_context_list)

        # 同步性测试

        Sync_list, C_list = ST(local_index_list, local_context_list)  # Sync_list= [True, True, True]

        # C_list= [['schedule', 'time'], ['favorite', 'favourite'], ['evening', 'night']]
        #         print("*"*30, "Sync_list=", Sync_list, "\n", "*"*30, "C_list=", C_list)

        Substitutable_list = [False] * len(Sync_list)
        copy_local_context_list = copy.deepcopy(local_context_list)
        copy_local_index_list = copy.deepcopy(local_index_list)
        for idxx, (Sync, C, (idx, target_token), local_context, local_index) in enumerate(
                zip(Sync_list, C_list, target_token_list, copy_local_context_list, copy_local_index_list)):
            if (Sync == True) and (target_token in C):
                Substitutable_list[idxx] = True

        # 判断Substitutable做最后的处理
        # print("", Substitutable_list)
        # print(rest_index_list)
        # print(C_list)
        assert len(Substitutable_list) == len(rest_index_list) == len(C_list)
        for substitutable, (idx, _), C in zip(Substitutable_list, rest_index_list, C_list):
            if substitutable is True:
                #  Fench one bit signal that has been embed in m
                m_rest = m[idx][count[idx]]
                #                 print("C[m_rest] = ", C[m_rest])
                #                 m_to_save[idx].append(m_rest)   # m_to_save= [[], [], [], [], []]
                m_to_save[idx] = m_to_save[idx] + [m_rest]  # 得到每句话最终的保存的二进制序列
                #  Replace t_index in S_w with word in C via Eq(7)
                #                 print("index_list[idx]=", index_list[idx])
                S_w_list[idx][1][index_list[idx] - 1] = C[m_rest]
                print("FC=", C)
                print("水印信号 m_rest=", m_rest)
                #                 print("Replace t_index in S_w with word in C via Eq(7):S_w_list[idx]=", S_w_list[idx])

                # 保存替换词集
                FC_to_save[idx] = FC_to_save[idx] + [(index_list[idx], C, m_rest)]

                count[idx] += 1

                latest_embed_index_list[idx] = index_list[idx]
                index_list[idx] = index_list[idx] + step
            else:
                index_list[idx] = index_list[idx] + 1
    #         print("!!"*30, "index_list=", index_list, "\nlatest_embed_index_list=", latest_embed_index_list)

    #     print("S_w_list1212=", S_w_list)
    # 将嵌入后的句子最处理连成完整的句子
    for idx, S_w, length in S_w_list:
        S_w_list[idx] = ' '.join(S_w)

    #     print("S_w_list1111111=", S_w_list)

    return S_w_list, m_to_save, FC_to_save

    # **************************************************水印提取*********************************************************************


def watermark_extraction(index_list, sentences_list):
    """
    水印提取过程
    参数说明：sen: 一个句子
    返回值：S_w: 水印提取
    """
    length = len(index_list)
    sen_token_list = [word_tokenize(sentence) for sentence in
                      sentences_list]  # sen_token_list= [['He', 'watches', 'his', 'favorite', 'show', 'every', 'night', 'on', 'time', '.'], ['He', 'watches', 'his', 'favorite', 'show', '.']]
    print("sen_token_list=", sen_token_list)

    # ****************************************************算法二**********************************************
    m = [] + [
        [0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1,
         1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1]] * length  # 水印二进制位序列

    m = [] + [[]] * length  # 水印二进制位序列  m= [[], [], [], [], []]
    count = [] + [0] * length  # 服务于水印二进制位序列的取值 [0, 0, 0, ]
    FC_to_save = [] + [[]] * length  # = [[], [], [], [], []]
    index_list = index_list  # 直接第二个词，没有从0索引开始
    latest_embed_index_list = [] + [0] * length
    max_length = max([len(sen) for sen in sen_token_list])  # 获取句子列表中的最长的那个句子的长度
    print("max_length=", max_length)

    for idx, sen_token in enumerate(sen_token_list):
        print("sen_token=", sen_token)
        sen_token_list[idx] = (idx, sen_token, len(sen_token))
    print("sen_token_list444444444444=", sen_token_list)

    # 获取最长句子在list中的位置
    max_length_idx = ([len(sen) for sen in sen_token_list]).index(max([len(sen) for sen in sen_token_list]))

    S_w_list = copy.deepcopy(sen_token_list)  # list
    #     print("&%^"*50, "S_w_list.type=", type(S_w_list), "S_w_list[0]=", S_w_list[0])
    copy_index_list = copy.deepcopy(index_list)  # 用来判断是否结束循环，列表元素全为0的时候结束循环

    while copy_index_list != [] + [0] * len(copy_index_list):
        # 判断index是否超出句子长度，超出的则证明已经嵌入完了，舍去
        rest_sentences = []
        rest_index_list = []  # (剩下的索引在index_list中的位置, 索引index)
        assert len(index_list) == len(S_w_list)
        for index, (idx, s_w, length) in zip(index_list, S_w_list):
            # index <= length-f说明嵌入过程还没有结束
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
            #             print("最后一步大治来啦----"*10)
            continue

        #         print("rest_sentences=", rest_sentences, "\nrest_index_list=", rest_index_list)

        assert len(rest_sentences) == len(rest_index_list)
        target_token_list = [(idx, s_w[index - 1]) for (idxx, index), (idx, s_w, length) in
                             zip(rest_index_list, rest_sentences)]  # (句子序号，目标词)

        #         print("target_token_list=", target_token_list)

        # 判断目标词中是否有无效词
        flag = True  # 不存在无效词为True
        for (idx, target_token) in target_token_list:
            #             print("target_token=", target_token, "target_next_token=", target_next_token)
            if (target_token.lower() in (RiskSet["punctuations"] + RiskSet["stopwords"] + RiskSet["subwords"])):
                index_list[idx] = index_list[idx] + 1
                flag = False
        # 如果存在无效词则重新while循环
        if flag == False:
            continue

        # 获取需要寻找替代词的句子集
        local_context_list = [s_w[:index] for (idxx, index), (idx, s_w, length) in zip(rest_index_list,
                                                                                       rest_sentences)]  ################################################################改了
        local_index_list = [index for idx, index in rest_index_list]
        #         print("*"*30, "local_index_list=", local_index_list, "\n", "*"*30, "local_context_list=", local_context_list)

        # 同步性测试
        Sync_list, C_list = ST(local_index_list, local_context_list)  # Sync_list= [True, True, True]
        # C_list= [['schedule', 'time'], ['favorite', 'favourite'], ['evening', 'night']]
        #         print("*"*30, "Sync_list=", Sync_list, "\n", "*"*30, "C_list=", C_list)

        Substitutable_list = [False] * len(Sync_list)
        copy_local_context_list = copy.deepcopy(local_context_list)
        copy_local_index_list = copy.deepcopy(local_index_list)
        for idxx, (Sync, C, (idx, target_token), local_context, local_index) in enumerate(
                zip(Sync_list, C_list, target_token_list, copy_local_context_list, copy_local_index_list)):
            if (Sync == True) and (target_token in C):
                Substitutable_list[idxx] = True

        # 判断Substitutable做最后的处理
        for substitutable, (idx, _), C in zip(Substitutable_list, rest_index_list, C_list):
            if substitutable is True:
                # print("#" * 20, "C=", C, "#" * 20)  # C= ['evening', 'night']
                # print("#" * 20, "sen_token_list[idx][1][index_list[idx]-1]=",
                #       sen_token_list[idx][1][index_list[idx] - 1], "#" * 20)
                # print("#" * 20, "sen_token_list[idx][1]=", sen_token_list[idx][1], "#" * 20)
                if sen_token_list[idx][1][index - 1] == C[0]:
                    m[idx] = m[idx] + [0]
                    # 保存替换词集
                    FC_to_save[idx] = FC_to_save[idx] + [(index_list[idx], C, 0)]
                elif sen_token_list[idx][1][index - 1] == C[1]:
                    m[idx] = m[idx] + [1]
                    # 保存替换词集
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
    #  # 嵌入
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

# # 一次处理的批数据大小
# length = 1
#
#
# if __name__ == "__main__":
#
#     # 数据加载
#     # file = ["Dracula",]
#     # file = ["Pride and Prejudice"]
#     # file = [ "Wuthering Heights"]
#     # file = ['ag-news-100']
#     # file = ['imdb-100']
#     file = ['wikitext-100']
#
#     # 水印嵌入
#     print("*"*40, "水印嵌入处理...", "*"*40)
#     for name in file:
#         novel_to_list_, novel_to_list_length_ = pickle.load(open("./MLM/novels_to_list/{}.pk1".format(name),"rb"))
#         novel_to_list_ = novel_to_list_[:]
# #         print("novel_to_list=", novel_to_list, "novel_to_list_length=", novel_to_list_length)
# #         print('sentence=', novel_to_list_[12:13])
#
#         for j in range(2): # Wuthering :0-68, 68-6699//50+1
#             novel_to_list = (novel_to_list_[0+j*50:] if j==5960//50 else novel_to_list_[0+j*50:0+(j+1)*50])
#             # novel_to_list[1] = ' '.join(word_tokenize(novel_to_list[1])[:160])
#             # novel_to_list = novel_to_list_[0+4*50+11:0+4*50+12]
#             novel_to_list_length = len(novel_to_list)
#     #         print("novel_to_list=", novel_to_list, "novel_to_list_length=", novel_to_list_length)
#             print("novel_to_list_length=", novel_to_list_length)
#
#             # 处理能被50整除的部分
#             S_w_list = []
#             m_to_save = []
#             FC_to_save = []
#             count = 0
#
#             for i in tqdm(range(novel_to_list_length//length)):
#                 count += 1
#                 print("{} -> {} dssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssss".format(count, name))
#                 index_list = [] + [2] * length   # [2, 2, 2, ]
#     #             index_list = [2, 7]
#                 sentences_list = novel_to_list[i*length:(i+1)*length]
#                 sentences_list = [' '.join(word_tokenize(sen)) for sen in sentences_list]
#                 print('sentences_list########################################=', sentences_list)
#
#                 # 水印嵌入
#                 S_w_list_, m_to_save_, FC_to_save_ = watermark_embedding(index_list, sentences_list)
#                 for S_w, m_to_save_item_, FC_to_save_item_ in zip(S_w_list_, m_to_save_, FC_to_save_):
#                     S_w_list.append(S_w)
#                     m_to_save.append(m_to_save_item_)
#                     FC_to_save.append(FC_to_save_item_)
#
#             # 处理整除后的余数部分
#             num = novel_to_list_length%length
#             print("num======================================================", num)
#             if num != 0:
#                 index_list = [] + [2] * num
#                 sentences_list = novel_to_list[novel_to_list_length - num:]
#                 # 水印嵌入
#                 S_w_list_, m_to_save_, FC_to_save_ = watermark_embedding(index_list, sentences_list)
#                 for S_w, m_to_save_item_, FC_to_save_item_ in zip(S_w_list_, m_to_save_, FC_to_save_):
#     #                 print("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
#                     S_w_list.append(S_w)
#                     m_to_save.append(m_to_save_item_)
#                     FC_to_save.append(FC_to_save_item_)
#     #         print("len(S_w_list_)=", len(S_w_list_),  "\nlen(novel_to_list)=", len(novel_to_list), "\nlen(m_to_save)=", len(m_to_save), "\nlen(FC_to_save)=", len(FC_to_save), "\nlen(FC_to_save_)=", len(FC_to_save_), "\nlen(m_to_save_)=", len(m_to_save_))  #10
#             # 整理数据
#             all_sample = []
#             for s, s_w, m_to_save_item, FC_to_save_item in zip(novel_to_list, S_w_list, m_to_save, FC_to_save):
#     #             print("进来了")
#                 all_sample.append((s, s_w, m_to_save_item, FC_to_save_item))
#
#             print("*"*30, "all_sample=", all_sample, "\nlength_sample=", len(all_sample))
#
#             assert len(S_w_list) == novel_to_list_length
#             # 将嵌入水印后的句子保存pk1
#             root_path = "./MLM/new_folder_Multi_no_substitutable_no_step_reversible_reversible-"+name.split(' ')[0]
#             print("正在将{}写入到pk1中...".format(name))
#             if not os.path.exists(root_path):
#                 os.makedirs(root_path, exist_ok=True)
#             name_ = name + "_watermark-" + str(j)
#             pickle.dump((novel_to_list, S_w_list, m_to_save, FC_to_save, all_sample, len(S_w_list)), open(root_path + "/{}-no-all-seed.pk1".format(name_), "wb"))
#             print(root_path + "  {}写入结束\n".format(name_))
