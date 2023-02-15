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

from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

from pattern.en import conjugate, lemma, lexeme

import pdb

from pathlib import Path
# import openpyxl
from nltk.corpus import stopwords

from nltk.stem import PorterStemmer  

from fairseq.models.transformer import TransformerModel

from nltk.tokenize import sent_tokenize, word_tokenize

from transformers import AutoTokenizer, AutoModelWithLMHead, AutoModelForSequenceClassification

from nltk.stem import PorterStemmer
ps = PorterStemmer()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


from bart.bart_score import BARTScorer

bart_scorer = BARTScorer(device=device, checkpoint='bart/bart-large-cnn')   
bart_scorer.load(path='bart/bart-large-cnn/parabank2/bart.pth')

bleurt_tokenizer = AutoTokenizer.from_pretrained("bleurt-large-512")
bleurt_model = AutoModelForSequenceClassification.from_pretrained("bleurt-large-512").to(device)
bleurt_model.eval()


def extract_substitute(output_sentences, output_scores, original_sentence, complex_word, method='bartscore', weight_bart=1.00, weight_original=0.02, weight_bleurt=1.00):
    original_words = nltk.word_tokenize(original_sentence) 

    index_of_complex_word = -1 

    if complex_word not in original_words:  
        i = 0
        for word in original_words:
            if complex_word == word.lower():
                index_of_complex_word = i
                break
            i += 1
    else:
        index_of_complex_word = original_words.index(complex_word)  
    
    if index_of_complex_word == -1:
        print("******************no found the complex word*****************")
        return [],[]

    
    len_original_words = len(original_words) 

    
    context = original_words[:] 
    context = " ".join([word for word in context]) 


    context = (context,index_of_complex_word)

    if output_sentences[0].find('<unk>'):  
        for i in range(len(output_sentences)):
            tran = output_sentences[i].replace('<unk> ', '')
            output_sentences[i] = tran

    complex_stem = ps.stem(complex_word)
    not_candi = set(['<unk>', "-", "``","*", "\"", 'the', 'with', 'of', 'a', 'an' , 'for' , 'in', 'to', 'but-but', 'populace'])  #  'it', 'me', 'is', 'it', 'on', 'of', 'date', 'a',
    not_candi.add(complex_stem)
    not_candi.add(complex_word)
    not_candi.add(complex_word+'s') 
    for alias in ['inf', '1sg', '2sg', '3sg', 'pl', 'part', 'p', '1sgp', '2sgp', '3sgp', 'ppl', 'ppart']:
        not_candi.add(conjugate(complex_word, alias))
    
    substitutes = []
    substitutes_scores = []
    
    sent_ind = -1
    
    substitutes.append(complex_word)
    substitutes_scores.append(1)
    

    # print('output_sentences=', output_sentences)
    for sentence in output_sentences:  
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

        candi_stem = ps.stem(candi)   
        candi_common = conjugate(candi, 'inf')
        if candi_stem in not_candi or candi in not_candi or candi_common in not_candi:
            continue
        # print('candi=', candi, '\nsent_ind=', sent_ind)
        if candi not in substitutes:
            substitutes.append(candi)
            substitutes_scores.append(output_scores[0][sent_ind])

    if len(substitutes)>0: 
        if method == 'bertscore':
            scores = substitutes_BertScore(context, complex_word, substitutes) 
        elif method == 'bartscore':
            bart_scores, bleurt_scores = substitutes_BartScore(context, complex_word, substitutes)
        final_scores = []

        for i in range(len(substitutes_scores)):
            score = weight_bart*bart_scores[i] + weight_original*substitutes_scores[i]+ weight_bleurt*bleurt_scores[i] 
            final_scores.append(score)

        rank_final_scores = sorted(final_scores, reverse=True)
        for score in rank_final_scores:
            print(score, '-->', substitutes[final_scores.index(score)])
        rank_final_substitutes = [substitutes[final_scores.index(v)] for v in rank_final_scores]

        return rank_final_substitutes, rank_final_scores

    return [],[]





def substitutes_BartScore(context, target, substitutes):
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
    scores_ = bart_scorer.score(refs, cands, batch_size=20)
    
    scores = torch.tensor(scores_)
    score_bart = scores.detach().numpy()

    with torch.no_grad():
        scores_bleurt = bleurt_model(**bleurt_tokenizer(refs, cands, return_tensors='pt', padding=True).to(device))[0].squeeze()
    scores_bleurt = scores_bleurt.cpu().detach().numpy()
    scores_bleurt = scores_bleurt.tolist()
    if isinstance(scores_bleurt,float):
       scores_bleurt = [] + [scores_bleurt]

    score_bart = score_bart.tolist()
    if isinstance(score_bart,float):
       score_bart = [] + [score_bart]
    return score_bart, scores_bleurt



    


def lexicalSubstitute(model, sentence, complex_word, complex_word_index,  beam):

    sentence_list = word_tokenize(sentence)
    index_complex = complex_word_index

    prefix = ""  

    if(index_complex != -1):
        prefix_list = sentence_list[0:index_complex-1]
        prefix = ' '.join(prefix_list)
        print('prefix=', prefix)
    else:
        sentence = sentence.lower()

        return lexicalSubstitute(model, sentence, complex_word,  beam)

    prefix_tokens = model.encode(prefix) 
    prefix_tokens = prefix_tokens[:-1].view(1,-1) 
    complex_tokens = model.encode(complex_word) 
    sentence_tokens = model.encode(sentence)   
    attn_len = len(prefix_tokens[0])+len(complex_tokens)-1  
    
    sentence_tokens = sentence_tokens.to(device)
    prefix_tokens = prefix_tokens.to(device)
  
    outputs,pre_scores = model.generate2(sentence_tokens, beam=beam, prefix_tokens=prefix_tokens, attn_len=attn_len)  
    two = [model.decode(x['tokens']) for x in outputs]

    output_sentences = [model.decode(x['tokens']) for x in outputs]    

    rank_final_substitutes, rank_final_scores = extract_substitute(output_sentences, pre_scores, sentence, complex_word)  

    return rank_final_substitutes, rank_final_scores  

   

if __name__ == "__main__":
    en2en = TransformerModel.from_pretrained('checkpoints/para/transformer/', checkpoint_file='checkpoint_best.pt', bpe='subword_nmt', bpe_codes='checkpoints/para/transformer/codes.40000.bpe.en')
    en2en.to(device)

    rank_final_substitutes, rank_final_scores = lexicalSubstitute(*sys.argv[1:])

    
    print("rank_final_substitutes=", rank_final_substitutes, '\nrank_final_scores=', rank_final_scores)
    
    

