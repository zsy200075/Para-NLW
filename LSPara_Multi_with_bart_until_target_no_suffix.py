#!/usr/bin/python
# -*- coding: UTF-8 -*-



from __future__ import absolute_import, division, print_function

import argparse
import csv
import logging
import os
import random
import math
import sys
import re

from sklearn.metrics.pairwise import cosine_similarity as cosine

import numpy as np
import torch
import torch.nn as nn
import nltk
import sys

from nltk.stem import WordNetLemmatizer # 用来提取次的词干或者词根

lemmatizer = WordNetLemmatizer()

from pattern.en import conjugate, lemma, lexeme

import pdb

from pathlib import Path
# import openpyxl
from nltk.corpus import stopwords

from nltk.stem import PorterStemmer   # 用来提取次的词干或者词根

from fairseq.models.transformer import TransformerModel

from nltk.tokenize import sent_tokenize, word_tokenize  # 分句工具

from transformers import AutoTokenizer, AutoModelWithLMHead, AutoModelForSequenceClassification

from nltk.stem import PorterStemmer
ps = PorterStemmer()

# from bert_score.scorer import BERTScorer ##################################################################----------bertscore
# scorer = BERTScorer(lang="en", rescale_with_baseline=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


from bart.bart_score import BARTScorer

bart_scorer = BARTScorer(device=device, checkpoint='bart/bart-large-cnn')    #################################################################----------bartscore
bart_scorer.load(path='bart/bart-large-cnn/parabank2/bart.pth')

bleurt_tokenizer = AutoTokenizer.from_pretrained("bleurt-large-512")
bleurt_model = AutoModelForSequenceClassification.from_pretrained("bleurt-large-512").to(device)
bleurt_model.eval()




def extract_substitute(output_sentences, output_scores, original_sentence, complex_word, method='bartscore', weight_bart=1.00, weight_original=0.02, weight_bleurt=1.00):
    """
    # 提取替代词， 方法二：考虑了所要代替词的后缀问题
    输入值:
        output_sentences --> 提取的替代词句子 ["The books were customized to the TV series.", "The books were tailored to the TV series.", "The books were modified into a TV series."]
        original_sentence --> 原始句子 "The books were adapted to the TV series."
        complex_word --> 目标词，复杂词 'adapted'
    输出值:
        bertscore_substitutes --> 过滤后的替代词集
        ranking_bertscore_substitutes --> 排序过滤后的替代词集
    """
    original_words = nltk.word_tokenize(original_sentence)  # 对原句子进行词单位的分词

    index_of_complex_word = -1   # 目标词的位置索引index

    if complex_word not in original_words:  
        i = 0
        for word in original_words:
            if complex_word == word.lower():
                index_of_complex_word = i
                break
            i += 1
    else:
        index_of_complex_word = original_words.index(complex_word)  # 目标词的位置索引index
    
    if index_of_complex_word == -1:
        print("******************no found the complex word*****************")
        return [],[]

    
    len_original_words = len(original_words)  # 获取原句子的词数

    
    context = original_words[:]  ##############################################################################
    context = " ".join([word for word in context])  # 将列表片段拼接成句子片段


    context = (context,index_of_complex_word)

    if output_sentences[0].find('<unk>'):  # 去除'<unk>'

        for i in range(len(output_sentences)):
            tran = output_sentences[i].replace('<unk> ', '')
            output_sentences[i] = tran

    complex_stem = ps.stem(complex_word)
    # print('complex_word=', complex_word)
    not_candi = set(['<unk>', "-", "``","*", "\"", 'the', 'with', 'of', 'a', 'an' , 'for' , 'in', 'to', 'but-but', 'populace'])  #  'it', 'me', 'is', 'it', 'on', 'of', 'date', 'a',
    # not_candi = set([complex_word, ])
    not_candi.add(complex_stem)
    not_candi.add(complex_word)
    not_candi.add(complex_word+'s') #####################################################################################################################################################################最新改动
    for alias in ['inf', '1sg', '2sg', '3sg', 'pl', 'part', 'p', '1sgp', '2sgp', '3sgp', 'ppl', 'ppart']:
        not_candi.add(conjugate(complex_word, alias))
    # print('not_candi=====================================', not_candi)
    
    # 对后缀的处理
    substitutes = []
    substitutes_scores = []
    
    sent_ind = -1
    
    substitutes.append(complex_word) # 容许复杂词自身%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    substitutes_scores.append(1)
    

    # print('output_sentences=', output_sentences)
    for sentence in output_sentences:  # 获取满足后缀词集的替代词列表
        sent_ind += 1
        sentence_list = nltk.word_tokenize(sentence)
        if len(sentence_list) < len(original_words):
            continue

        
        if len(sentence) < 3:
            continue
        words = nltk.word_tokenize(sentence)
        if index_of_complex_word>=len(words):
            continue
        candi = words[index_of_complex_word].lower().strip()
        if candi[-1] == '.' or candi[-1] == ',' or candi[-1] == '?' \
                or candi[-1] == '!' or candi[-1] == '-' or candi[-1] == '——':
            candi = candi[:-1]

        candi_stem = ps.stem(candi)   # 用来提取目标词的词干或者词根 -->  例子：'adapted'-->complex_stem=="adapt"
        candi_common = conjugate(candi, 'inf')
        if candi_stem in not_candi or candi in not_candi or candi_common in not_candi:
            continue
        # print('candi=', candi, '\nsent_ind=', sent_ind)
        if candi not in substitutes:
            substitutes.append(candi)
            substitutes_scores.append(output_scores[0][sent_ind])
          
        
        
    print("未经过bert_score的筛选后的替代词extract_substitute_substitutes=", substitutes)
   
    
    
    if len(substitutes)>0:  # 对满足后缀要求的替代词进行bert_score打分
        
        if method == 'bertscore':
            scores = substitutes_BertScore(context, complex_word, substitutes)  # 返回bert_score中的f1
        elif method == 'bartscore':
            bart_scores, bleurt_scores = substitutes_BartScore(context, complex_word, substitutes)
        
        #print(substitutes)
        final_scores = []
        # print('bart_scores=', type(bart_scores), '\nsubstitutes_scores=', type(substitutes_scores), '\nbleurt_scores=',
        #       type(bleurt_scores))
        # print('bart_scores=', bart_scores, '\nsubstitutes_scores=', substitutes_scores, '\nbleurt_scores=', bleurt_scores)


        for i in range(len(substitutes_scores)):
            score = weight_bart*bart_scores[i] + weight_original*substitutes_scores[i]+ weight_bleurt*bleurt_scores[i] 
            # print(substitutes[i], '-->', score, ' type=', type(score))
            final_scores.append(score)
        #pdb.set_trace()

        # print('substitutes=', substitutes, '\nfinal_scores=', final_scores, '\n\n')
        rank_final_scores = sorted(final_scores, reverse=True)
        for score in rank_final_scores:
            print(score, '-->', substitutes[final_scores.index(score)])
        rank_final_substitutes = [substitutes[final_scores.index(v)] for v in rank_final_scores]

        return rank_final_substitutes, rank_final_scores

    return [],[]





def substitutes_BartScore(context, target, substitutes):
    """
    计算bert_score分数
    输入值:
        context--> 原句子所截取的片段与目标词在片段中的位置（context = (context,index_of_complex_in_context)），通常是目标词前四后四的句子片段 --> The books were adapted to the TV series, 3
        target--> 目标词，索要替代的词  --> adapted
        substitutes--> 经过所生成的替代句的前后缀筛选过后的的替换词集  --> [' ', ' ']
    返回值:
        F1--> 原句片段与替代句子片段之间的bert_score分数
    """
    refs = []
    cands = []
    target_id = context[1]
    sent = context[0]

    words = sent.split(" ")
    for sub in substitutes:
        refs.append(sent)
        
        new_sent = ""
        
        for i in range(len(words)):
            if i==target_id:
                new_sent += sub + " "
            else:
                new_sent += words[i] + " "
        cands.append(new_sent.strip())
    
    # print('refs=', refs, '\ncands=', cands)
    scores_ = bart_scorer.score(refs, cands, batch_size=20)
    
#     print('scores_=', scores_, '\ntype=', type(scores_))
    scores = torch.tensor(scores_)
# #     print('type=', type(scores))
#     scores = nn.Softmax(dim=0)(scores)
    score_bart = scores.detach().numpy()
    # for ref, cand, word in zip(refs, cands, substitutes, ):
        # print('\nref=', ref, '\ncand=', cand, '\nword=', word, '\n\n')

    with torch.no_grad():
        scores_bleurt = bleurt_model(**bleurt_tokenizer(refs, cands, return_tensors='pt', padding=True).to(device))[0].squeeze()
    scores_bleurt = scores_bleurt.cpu().detach().numpy()
    scores_bleurt = scores_bleurt.tolist()

    # print('type(xxx)=', type(scores_bleurt))
    # print('isinstance(scores_bleurt,float)=', isinstance(scores_bleurt,float))
    if isinstance(scores_bleurt,float):
       scores_bleurt = [] + [scores_bleurt]

    score_bart = score_bart.tolist()
    if isinstance(score_bart,float):
       score_bart = [] + [score_bart]
       
    # print('score_bart=', score_bart, '\nscores_bleurt=', scores_bleurt)
    return score_bart, scores_bleurt



    


def lexicalSubstitute(model, sentence, complex_word, complex_word_index,  beam):#

    sentence_list = word_tokenize(sentence)
    # index_complex = sentence.find(complex_word)  # 这里的sentence是一句话相当于长字符串"The books were adapted to the TV series.",complex_word="adapted", -->index_complex=15从句子开头到adapted的首字母之前
    index_complex = complex_word_index

    prefix = ""  # 获取前缀

    if(index_complex != -1):
        prefix_list = sentence_list[0:index_complex-1]
        prefix = ' '.join(prefix_list)
        print('prefix=', prefix)
    else:
        #print("*************cannot find the complex word")
        #print(sentence)
        #print(complex_word)
        sentence = sentence.lower()

        return lexicalSubstitute(model, sentence, complex_word,  beam)

    prefix_tokens = model.encode(prefix)   # 将前缀词转化为token数字列表
    prefix_tokens = prefix_tokens[:-1].view(1,-1)  # 

    complex_tokens = model.encode(complex_word)  # 复杂词，目标词的数字token
#     print("lexicalSubstitute_complex_tokens=", complex_tokens)
    
    sentence_tokens = model.encode(sentence)   # 将整句话转化为数字token
#     print("lexicalSubstitute_sentence_tokens=", sentence_tokens)

    attn_len = len(prefix_tokens[0])+len(complex_tokens)-1  # 自注意机制
    
    sentence_tokens = sentence_tokens.to(device)
    prefix_tokens = prefix_tokens.to(device)
#     print("sentence_tokens.device=", sentence_tokens.device)

    # print('sentence_tokens=', sentence_tokens, '\nprefix_tokens=', prefix_tokens, '\nattn_len=', attn_len, '\n\n')
    outputs,pre_scores = model.generate2(sentence_tokens, beam=beam, prefix_tokens=prefix_tokens, attn_len=attn_len)   # generate2返回的outputs是关于beam句话的详细信息的列表，其中每句话的详细信息的表现形式是词典，
    # pre_scores = pre_scores.cpu().numpy()
#     print('type=', type(pre_scores))
#     print(outputs, '\npre_scores=', pre_scores)

    two = [model.decode(x['tokens']) for x in outputs]

    # for sen in two:
    #     print('sen######################3=', sen)
    # print("\n\n")
    
   
    
    output_sentences = [model.decode(x['tokens']) for x in outputs]  # 词典中的‘token’就是指替换后的句子的token列表：'tokens': tensor([  19, 4261,   50, 4536,   70,    2]),，这里是将数字列表解码为句子：The books were customized to the TV series.
#     for sen in output_sentences:
#         print(sen)
        
        
    rank_final_substitutes, rank_final_scores = extract_substitute(output_sentences, pre_scores, sentence, complex_word)  # 提取替代词

    # print("rank_final_substitutes=", rank_final_substitutes)
    return rank_final_substitutes, rank_final_scores   #


   

if __name__ == "__main__":
# # #     main()
    en2en = TransformerModel.from_pretrained('checkpoints/para/transformer/', checkpoint_file='checkpoint_best.pt', bpe='subword_nmt', bpe_codes='checkpoints/para/transformer/codes.40000.bpe.en')
    en2en.to(device)

    rank_final_substitutes, rank_final_scores = lexicalSubstitute(*sys.argv[1:])# (en2en, 'your sentence', 'your target word', 3, 20, )

    
    print("rank_final_substitutes=", rank_final_substitutes, '\nrank_final_scores=', rank_final_scores)
    
    

