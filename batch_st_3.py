import torch
import re
import copy
import numpy as np
import pickle

from fairseq.models.transformer import TransformerModel

import nltk
from nltk.corpus import wordnet as wn
from nltk.tokenize import sent_tokenize, word_tokenize  # 分句工具
from pattern.en import conjugate, lemma, lexeme

import LSPara
import LSPara_bart_until_target_no_suffix
import LSPara_Multi_with_bart_until_target_no_suffix

# config
beam = K = 20

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 加载Paraphraser模型
en2en = TransformerModel.from_pretrained(
    'checkpoints/para/transformer/',
    checkpoint_file='checkpoint_best.pt',
    bpe='subword_nmt',
    bpe_codes='checkpoints/para/transformer/codes.40000.bpe.en'
)


def get_FC(index_list, sen_token_list):
    """
    获取筛选后的候选词集
    参数说明：index: 需要替换的词（需要Mask的词）在句子中的是第几个词, 列表
              sen_token_list: 需要处理的句子token列表
    """
    index = index_list[0]
    sen_token = sen_token_list[0]
    target_token = sen_token[index - 1]
    # print('index=', index, '\nsen_token=', sen_token, '\n\n')
    # print("target_token=", target_token)
    # 数据处理
    sen_token = sen_token  # ['The', 'capcity', 'of', 'France', 'is', 'Paris', '.']
    #     print("index_list=", index_list, "\nsen_token_list=", sen_token_list)
    sentence = ' '.join(sen_token)
    print('sentence=', sentence, '\ntarget_token=', target_token, '\nindex=', index)

    FC_list = []
    FC_score_list = []

    # if target_token == '':

    rank_final_substitutes, rank_final_scores = LSPara_Multi_with_bart_until_target_no_suffix.lexicalSubstitute(
        en2en.to(device), sentence, target_token, index, beam)  #
    if len(rank_final_substitutes) >= 2:
        FC_list.append(rank_final_substitutes[:2])
        FC_score_list.append(rank_final_scores[:2])
    else:
        FC_list.append(rank_final_substitutes)
        FC_score_list.append(rank_final_scores)

    return FC_list, FC_score_list  # [['favorite', 'favourite']]


fileter_list = ['-', '_', '*', '(', ')', '{', '}', '[', ']', '《', '》', '@', '#', "$", "%", "^", "&", "~", \
                "`", 'd', 't', 's', 'm', 'al', 'unk', '<unk>', '', "''", '``', ]


# 'the', 'with', 'of', 'a', 'an' , \
#                 'for' , 'in', 'are', 'is', 'who', 'whom', 'off', 'to', 'but-but', 'heat-', 'by', 'into'
def ST(index_list, sen_token_list, is_test=False):
    """
    Synchronicity Test
    参数说明：index: 需要替换的词（需要Mask的词）在句子中的是第几个词
              sen_token_list: 需要处理的句子token列表
    """

    assert len(index_list) == len(sen_token_list)

    FC_list, FC_score_list = get_FC(index_list, sen_token_list)  # FC_list= [['favorite', 'favourite']]

    # 原来是准备批量数据处理的，最终只能batch_size=1, 所以最终的批量输出始终只有1个数据
    FC = FC_list[0]
    FC_score = FC_score_list[0]
    # print('FC=', FC, '\nFC_score=', FC_score, )

    Sync_list = [False] * 1
    C_list = [[]] * 1

    # 判断FC中是否有标点符号那些，如果有也直接排除掉
    flag_ = False
    for fc in FC:
        if fc in fileter_list:
            flag_ = True
            break

    if len(FC) < 2 or (flag_ == True) or len(FC[-1]) > 20:
        return Sync_list, C_list

    else:
        if is_test:
            # 获取除去目标词自身的第二个分值最大的词, 并对其进行可逆性测试
            # print('进来啦')
            sub_word = FC[-1]
            sub_score = FC_score[-1]  # score(c/t)  目标词t，替代词c
            sub_sen_token = copy.deepcopy(sen_token_list[0])
            sub_index = index_list[0]
            sub_sen_token[sub_index - 1] = sub_word
            sub_index_list = [sub_index]
            sub_sen_token_list = [sub_sen_token]
            # print('sub_index_list=', sub_index_list, '\nsub_sen_token_list=', sub_sen_token_list)
            # 测试可逆性
            sub_FC_list, sub_FC_score_list = get_FC(sub_index_list, sub_sen_token_list)
            sub_FC = sub_FC_list[0]
            sub_FC_score = sub_FC_score_list[0]
            print('sub_FC=', sub_FC, '\nsub_FC_score=', sub_FC_score)

            if len(sub_FC) == 2:
                score = sub_FC_score[-1]  # score(t/c)  目标词t，替代词c，由词c得到t的分数
                print('大治进来啦')
                print('{}->sub_score:{} | {}->score:{}'.format(FC, sub_score, sub_FC, score), '\n')
                if sub_FC[-1] == FC[0]:  # 既要满足top(top(t))=t，也要满足score(c/t) > score(t/c)
                    Sync_list[0] = True
                    C_list[0] = sorted(FC)

        else:
            if len(FC) == 2:
                Sync_list[0] = True
                C_list[0] = sorted(FC)

            # print('FC=', sorted(FC), '\nFC_score=', FC_score, )

        return Sync_list, C_list


if __name__ == "__main__":
    index_list = [4]
    sen_token_list = [['He', 'watches', 'his', 'favorite', 'show', 'every', 'evening', 'on', 'schedule', '.'], ]
    # sen_token_list = [['resulting', 'in', 'a', 'population', 'decline', 'as', 'workers', 'left', 'for', 'other',
    # 'areas'], ]

    Sync_list, C_list = ST(index_list=index_list, sen_token_list=sen_token_list)
