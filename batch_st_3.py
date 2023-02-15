import torch
import re
import copy
import numpy as np
import pickle

from fairseq.models.transformer import TransformerModel

import nltk
from nltk.corpus import wordnet as wn
from nltk.tokenize import sent_tokenize, word_tokenize  
from pattern.en import conjugate, lemma, lexeme

import LSPara
import LSPara_bart_until_target_no_suffix
import LSPara_Multi_with_bart_until_target_no_suffix

beam = K = 20

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

en2en = TransformerModel.from_pretrained(
    'checkpoints/para/transformer/',
    checkpoint_file='checkpoint_best.pt',
    bpe='subword_nmt',
    bpe_codes='checkpoints/para/transformer/codes.40000.bpe.en'
)


def get_FC(index_list, sen_token_list):
    index = index_list[0]
    sen_token = sen_token_list[0]
    target_token = sen_token[index - 1]
    sen_token = sen_token 
    sentence = ' '.join(sen_token)

    FC_list = []
    FC_score_list = []

    rank_final_substitutes, rank_final_scores = LSPara_Multi_with_bart_until_target_no_suffix.lexicalSubstitute(
        en2en.to(device), sentence, target_token, index, beam) 
    if len(rank_final_substitutes) >= 2:
        FC_list.append(rank_final_substitutes[:2])
        FC_score_list.append(rank_final_scores[:2])
    else:
        FC_list.append(rank_final_substitutes)
        FC_score_list.append(rank_final_scores)

    return FC_list, FC_score_list 

fileter_list = ['-', '_', '*', '(', ')', '{', '}', '[', ']', '《', '》', '@', '#', "$", "%", "^", "&", "~", \
                "`", 'd', 't', 's', 'm', 'al', 'unk', '<unk>', '', "''", '``', ]


def ST(index_list, sen_token_list, is_test=False):
    assert len(index_list) == len(sen_token_list)
    FC_list, FC_score_list = get_FC(index_list, sen_token_list) 
    FC = FC_list[0]
    FC_score = FC_score_list[0]
    Sync_list = [False] * 1
    C_list = [[]] * 1
    flag_ = False
    for fc in FC:
        if fc in fileter_list:
            flag_ = True
            break
    if len(FC) < 2 or (flag_ == True) or len(FC[-1]) > 20:
        return Sync_list, C_list
    else:
        if is_test:
            sub_word = FC[-1]
            sub_score = FC_score[-1] 
            sub_sen_token = copy.deepcopy(sen_token_list[0])
            sub_index = index_list[0]
            sub_sen_token[sub_index - 1] = sub_word
            sub_index_list = [sub_index]
            sub_sen_token_list = [sub_sen_token]
            # print('sub_index_list=', sub_index_list, '\nsub_sen_token_list=', sub_sen_token_list)
            sub_FC_list, sub_FC_score_list = get_FC(sub_index_list, sub_sen_token_list)
            sub_FC = sub_FC_list[0]
            sub_FC_score = sub_FC_score_list[0]
            print('sub_FC=', sub_FC, '\nsub_FC_score=', sub_FC_score)
            if len(sub_FC) == 2:
                score = sub_FC_score[-1]
                print('{}->sub_score:{} | {}->score:{}'.format(FC, sub_score, sub_FC, score), '\n')
                if sub_FC[-1] == FC[0]:  
                    Sync_list[0] = True
                    C_list[0] = sorted(FC)
        else:
            if len(FC) == 2:
                Sync_list[0] = True
                C_list[0] = sorted(FC)

        return Sync_list, C_list


if __name__ == "__main__":
    index_list = [4]
    sen_token_list = [['He', 'watches', 'his', 'favorite', 'show', 'every', 'evening', 'on', 'schedule', '.'], ]
    # sen_token_list = [['resulting', 'in', 'a', 'population', 'decline', 'as', 'workers', 'left', 'for', 'other',
    # 'areas'], ]

    Sync_list, C_list = ST(index_list=index_list, sen_token_list=sen_token_list)
