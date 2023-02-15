import os
import pickle
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
import copy
import re

from transformers import AutoTokenizer, BertForSequenceClassification, BertConfig, \
    get_linear_schedule_with_warmup, BertPreTrainedModel, BertModel, BertForMaskedLM, BertTokenizer
from nltk.tokenize import sent_tokenize, word_tokenize   # 分句工具

from bert_gen import bert_gen   # 加载模型


# Paraphraser_watermark_files   novels_to_list           './MLM/novels_to_list/Pride and Prejudice_watermark-500-1500.pk1'
# Dracula    Pride and Prejudice
def count_candidate_set(all_sample):
    """
    统计替代词数量和完全没有改变的句子数量
    """
    count_ls = 0
    count_kong = 0
    count_one = 0
    count_two = 0
    count_three = 0
    count_four = 0
  
    count_5 = 0
    count_6 = 0
    count_7 = 0
    count_8 = 0
    count_9 = 0
 
    count_10 = 0
    count_11 = 0
    count_12 = 0
    count_13 = 0
    count_14 = 0
    count_15, count_16, count_17, count_18, count_19, count_20, count_21, count_22, count_23, count_24, count_25 = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    count_26, count_27, count_28, count_29, count_30, count_31, count_32 = 0, 0, 0, 0, 0, 0, 0 
    count_33, count_34, count_35, count_36, count_37, count_38, count_39, count_40 = 0, 0, 0, 0, 0, 0, 0, 0 
    for (_, _, m_to_save_item, _) in all_sample:
        print("\nls=", _)
        count_ls += len(m_to_save_item)
        if len(m_to_save_item) == 0:
            count_kong += 1
        if len(m_to_save_item) == 1:
            count_one += 1
        if len(m_to_save_item) == 2:
            count_two += 1
        if len(m_to_save_item) == 3:
            count_three += 1
        if len(m_to_save_item) == 4:
            count_four += 1
              
        if len(m_to_save_item) == 5:
            count_5 += 1
        if len(m_to_save_item) == 6:
            count_6 += 1
        if len(m_to_save_item) == 7:
            count_7 += 1
        if len(m_to_save_item) == 8:
            count_8 += 1
        if len(m_to_save_item) == 9:
            count_9 += 1
            
        if len(m_to_save_item) == 10:
            count_10 += 1
        if len(m_to_save_item) == 11:
            count_11 += 1
        if len(m_to_save_item) == 12:
            count_12 += 1
        if len(m_to_save_item) == 13:
            count_13 += 1
        if len(m_to_save_item) == 14:
            count_14 += 1
            
        if len(m_to_save_item) == 15:
            count_15 += 1
        if len(m_to_save_item) == 16:
            count_16 += 1
        if len(m_to_save_item) == 17:
            count_17 += 1
        if len(m_to_save_item) == 18:
            count_18 += 1
        if len(m_to_save_item) == 19:
            count_19 += 1
            
        if len(m_to_save_item) == 20:
            count_20 += 1
        if len(m_to_save_item) == 21:
            count_21 += 1
        if len(m_to_save_item) == 22:
            count_22 += 1
        if len(m_to_save_item) == 23:
            count_23 += 1
        if len(m_to_save_item) == 24:
            count_24 += 1
        if len(m_to_save_item) == 25:
            count_25 += 1
            
        if len(m_to_save_item) == 26:
            count_26 += 1
        if len(m_to_save_item) == 27:
            count_27 += 1
        if len(m_to_save_item) == 28:
            count_28 += 1
        if len(m_to_save_item) == 29:
            count_29 += 1
        if len(m_to_save_item) == 30:
            count_30 += 1
        if len(m_to_save_item) == 31:
            count_31 += 1
        if len(m_to_save_item) == 32:
            count_32 += 1
            
        if len(m_to_save_item) == 33:
            count_33 += 1
        if len(m_to_save_item) == 34:
            count_34 += 1
        if len(m_to_save_item) == 35:
            count_35 += 1
        if len(m_to_save_item) == 36:
            count_36 += 1
        if len(m_to_save_item) == 37:
            count_37 += 1
        if len(m_to_save_item) == 38:
            count_38 += 1
        if len(m_to_save_item) == 39:
            count_39 += 1
        if len(m_to_save_item) == 40:
            count_40 += 1

    print("{} count_ls= ".format(file_path), count_ls, "\n{} count_kong= ".format(file_path), count_kong, "\n{} count_one= ".format(file_path), count_one, "\n{} count_two= ".format(file_path), count_two, "\n{} count_three= ".format(file_path), count_three, "\n{} count_four= ".format(file_path), count_four,  "\n{} count_5= ".format(file_path), count_5,  "\n{} count_6= ".format(file_path), count_6, "\n{} count_7= ".format(file_path), count_7, "\n{} count_8= ".format(file_path), count_8, "\n{} count_9= ".format(file_path), count_9, "\n{} count_10= ".format(file_path), count_10, "\n{} count_11= ".format(file_path), count_11, "\n{} count_12= ".format(file_path), count_12, "\n{} count_13= ".format(file_path), count_13, "\n{} count_14= ".format(file_path), count_14
         ,  "\n{} count_15= ".format(file_path), count_15, "\n{} count_16= ".format(file_path), count_16, "\n{} count_17= ".format(file_path), count_17, "\n{} count_18= ".format(file_path), count_18, "\n{} count_19= ".format(file_path), count_19, "\n{} count_20= ".format(file_path), count_20, "\n{} count_21= ".format(file_path), count_21, "\n{} count_22= ".format(file_path), count_22, "\n{} count_23= ".format(file_path), count_23
         ,  "\n{} count_24= ".format(file_path), count_24, "\n{} count_25= ".format(file_path), count_25, "\n{} count_26= ".format(file_path), count_26, "\n{} count_27= ".format(file_path), count_27, "\n{} count_28= ".format(file_path), count_28, "\n{} count_29= ".format(file_path), count_29, "\n{} count_30= ".format(file_path), count_30, "\n{} count_31= ".format(file_path), count_31, "\n{} count_32= ".format(file_path), count_32
         ,  "\n{} count_33= ".format(file_path), count_33, "\n{} count_34= ".format(file_path), count_34, "\n{} count_35= ".format(file_path), count_35, "\n{} count_36= ".format(file_path), count_36, "\n{} count_37= ".format(file_path), count_37, "\n{} count_38= ".format(file_path), count_38, "\n{} count_39= ".format(file_path), count_39, "\n{} count_40= ".format(file_path), count_40)
    
    num_samples = len(all_sample)   
    
    return count_ls, count_kong, count_one, count_two, count_three, count_four
    
def average_count(novel_to_list, count_ls, num_samples, fileter_door):
    novel_to_list=  novel_to_list

    assert len(novel_to_list) == num_samples
    print("\nnum_samples=", num_samples)
    # 预处理
    if fileter_door: 
        fileter = ['"', "\:", "\'", "\,", "\;", "\.", "\?", "\!", '\-', '\_', '\*', '\(', '\)', '\{', '\}', '\[', '\]',]   #   
        novel_to_list = [re.sub("|".join(fileter), ' ', item) for item in novel_to_list]
    word_list = [word_tokenize(sentence) for sentence in novel_to_list] 
    
    count = 0
    for item in word_list:
        count += len(item)
        
    average_sentence_word = count // len(novel_to_list)
    
    payload = round(count_ls / count, 3)

    return average_sentence_word, payload




# 获取bert的分词器， 仅仅用来对句子进行分词
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-cased", cache_dir='./pretrained_models/bert-base-cased',)   

# 获取bert词典
src_vocab_file = "./pretrained_models/bert-base-cased/6508e60ab3c1200bffa26c95f4b58ac6b6d95fba4db1f195f632fa3cd7bc64cc.437aa611e89f6fc6675a049d2b5545390adbc617e7d655286421c191d2be2791"
with open(src_vocab_file, 'r', encoding='utf-8') as f:
    lines = f.read().split('\n')
# print("num_list=",range(4))
# 获取src词典word2id
src_vocab = dict(zip(lines,list(range( bert_tokenizer.vocab_size))))
# with open("adc", 'w', encoding='utf-8') as f:
#       f.write(str(src_vocab))
# print("src_vocab=", src_vocab)
# 获取只有字符的列表
src_vocab_list = list(src_vocab)
# print("src_vocab_list[:10]=", src_vocab_list[:10])
# 获取id2word词典
inverse_src_vocab = dict(zip(src_vocab.values(),src_vocab.keys()))




def read_file_get_all_sample(file_path):
    """
    读取文件返回all_sample
    """
    S___list, S_w_list, m_to_save, FC_to_save, all_sample, length_ = pickle.load(open(file_path,"rb"))
#     all_sample = all_sample[:200]
    return all_sample


def text_recoverability(file_path, name_, t_r=True, fileter_door=True):
    file_path = file_path
#     tag = (file_path.split('/')[2]).split("_")[0]   # 'novels_to_list'  'Paraphraser_watermark_files'    './MLM/novels_to_list/Pride and Prejudice_watermark-500-1500.pk1'  file_path = './MLM/novels_to_list_random_m_pattern/Wuthering Heights_watermark-500-1500.pk1'
    tag = '_'.join((file_path.split('/')[2]).split('_')[:2])
#     tag = 'new_folder'
    name = ((file_path.split('/')[-1]).split(".")[0]).split('_')[0] + (((file_path.split('/')[-1]).split(".")[0]).split('_')[1]).split('mark')[-1]
#     tag = 'novels_to_list'  # 'novels_to_list'  'Paraphraser_watermark_files'    './MLM/novels_to_list/Pride and Prejudice_watermark-500-1500.pk1'
    
    

    S___list, S_w_list, m_to_save, FC_to_save, all_sample, length_ = pickle.load(open(file_path,"rb"))
#     all_sample = all_sample[2000:3000]
    novel_to_list, novel_to_list_length = pickle.load(open("./MLM/novels_to_list/{}.pk1".format(name_),"rb"))
#     novel_to_list = novel_to_list[2000:3000]
    num_samples = len(all_sample)
     
    count_ls, count_kong, count_one, count_two, count_three, count_four = count_candidate_set(all_sample)
    # 计算payload
    average_sentence_word, payload = average_count(novel_to_list, count_ls, num_samples, fileter_door)
    print("\naverage_sentence_word=", average_sentence_word, "\npayload=", payload)
    print("\nlen(all_sample)=", len(all_sample))
    
    bert_gen.eval()
    
    
    if t_r:
        all_sample_recovery = []
        count_recovery = 0
        for sample in tqdm(all_sample):
            (s, s_w, m_to_save_item, FC_to_save_item) = sample
            sentence = s
            sentence_w = s_w

            # 旧 run_watermark.py
            if tag == 'old_folder':
                sentence_word = bert_tokenizer.tokenize(sentence)  # 原始句子
    #             print("sentence_word=", sentence_word)
                sentence_w_word = bert_tokenizer.tokenize(sentence_w)

            # 新 run_watermark_2.py
            if tag == 'new_folder':
                sentence_word = word_tokenize(sentence)  # 原始句子
        #         print("sentence_word=", sentence_word)
                sentence_w_word = word_tokenize(sentence_w)  # 嵌入句

            candidate_set_recovery = []
            for item in FC_to_save_item:
                (idx, candidate_set, _) = item
                original_word = sentence_word[idx-1]
    #             print("\noriginal_word=", original_word)
                copy_sentence_w_word = copy.deepcopy(sentence_w_word)
    #             print("copy_sentence_w_word=", copy_sentence_w_word, "len(copy_sentence_w_word)=", len(copy_sentence_w_word))
                candidate_A, candidate_B = candidate_set

                # 数据预处理
                copy_sentence_w_word[idx-1] = '[MASK]'
                copy_sentence_w = " ".join(copy_sentence_w_word)
                input_tokenizer = bert_tokenizer(copy_sentence_w, padding=True, truncation=True, return_tensors='pt')
                input_2id = (input_tokenizer["input_ids"]).cuda()
                attention_mask = (input_tokenizer["attention_mask"]).cuda()  #在多数据自动补齐pad时pad位置为0
   
                # 获取Mask位置为1，其他都为0, 
                mask_ = (input_2id == bert_tokenizer.mask_token_id).cpu()
                mask_ = mask_.numpy()
                mask_ = mask_.astype(int)
    #             print("mask_=", mask_)  # mask_= [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]]
    #             print("type(mask_)=",type(mask_))   # type(mask_)= <class 'numpy.ndarray'>
                mask_ = torch.from_numpy(mask_)
                mask_.cuda()
    #             print("mask_=", mask_)   # mask_= tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

    #             break
                with torch.no_grad():
                    output = bert_gen(input_2id, mask_, attention_mask)
    #             print("output.shape=", output)
                output = output.view(-1)
    #             print("output.shape=", output.shape)
                output_list = output.detach().cpu().numpy()
    #             print("output_list=", output_list)      
    #             print("bert_tokenizer.vocab_size=", bert_tokenizer.vocab_size)  # 28996
                output_dict_id_2word = dict(zip(list(range( bert_tokenizer.vocab_size)), list(output_list)))
    #             print("output_dict_id_2word=", output_dict_id_2word)

                # 将替代词集to id并根据概率排序
                flag = False
                for candidate in candidate_set:
                    if candidate not in src_vocab_list:
                        flag = True

                if flag:
                    candidate_top = original_word
                    candidate_set_2id = None

                elif not flag:
                    candidate_set_2id = sorted([(candidate_set[i] ,output_dict_id_2word[src_vocab[candidate_set[i]]]) for i in range(len(candidate_set))], key=lambda x:x[-1], reverse=True)    # candidate_set_2id= [124, 1210]
                    candidate_top = candidate_set_2id[0][0]
    #             print("candidate_set_2id=", candidate_set_2id, "\ncandidate_top=", candidate_top, "\n")   # candidate_set_2id= [('3', 4.4750724), ('three', 9.898824)]

                candidate_set_recovery.append((idx, original_word, candidate_top))

            for item in candidate_set_recovery:

                if item[1] == item[2]:
    #                 print("item=", item)
                    count_recovery += 1 

            all_sample_recovery.append((s, candidate_set_recovery)) 

    #     print("\nall_sample_recovery=", all_sample_recovery, "\nlen(all_sample_recovery)=", len(all_sample_recovery))
        print("count_recovery=", count_recovery, "\n") 

        text_recoverability =  count_recovery/count_ls
        text_recoverability_tosave = 'percent: {:.2%}'.format(text_recoverability)
        print(text_recoverability_tosave) 
        # 保存
        if not os.path.exists("./MLM/text_recoverability"):
            os.makedirs("./MLM/text_recoverability", exist_ok=True)
        name_ = tag + "_" + name 
        print("{}正在写入".format(name_))
#         pickle.dump((all_sample_recovery, len(all_sample_recovery), text_recoverability_tosave), open("./MLM/text_recoverability/{}.pk1".format(name_), "wb"))
        print("{}写入结束".format(name_))
        
        return (all_sample_recovery, len(all_sample_recovery), text_recoverability_tosave)
        

        
# fileter_u = [ ]
fileter_u = [ 'his', 'hers', 'its', 'ours', 'yours', 'him', 'them', 'her', 'us', 'their', 'our', 'your', 'herself', 'himself', 'themselves', 'ourselves', 'itself', 'he', 'she', 'it', 'we', 'they', 'i',  'you',]
fileter_jieci = [ 'for', 'to' ,'the', 'away', 'from', 'on', 'upon', 'to', 'unto', 'by', 'into', 'in', 'of', 'off', 'with', 'at' , 'toward', 'd', 't', 's', 'm', 'al', 'into', 'away', 'off', 'upon', ]  # 
# 
# 计算u或者介词数量
def just_u(file_path, is_count_u=True, is_count_jieci=True):
    """
    统计替代词数量和完全没有改变的句子数量
    """
    file_path = file_path
    S___list, S_w_list, m_to_save, FC_to_save, all_sample, length_ = pickle.load(open(file_path,"rb"))

    
    new_all_sample_u = []
    count_u = 0
    if is_count_u:
#         print("count_u")
        for (s, s_w, m_to_save_item, FC_to_save_list) in all_sample:
            new_FC_to_save_list = []
    #         print("\nls=", FC_to_save_list)
            if FC_to_save_list != []:
                for item in FC_to_save_list:
                    for fc in item[1][:]:
                        if fc.lower() in fileter_u:
                            new_FC_to_save_list.append(item)
                            break

            new_all_sample_u.append((s, new_FC_to_save_list))
    
        # count
        for (s, new_FC_to_save_list) in new_all_sample_u:
            count_u += len(new_FC_to_save_list)
#         print("\ncount_u=", count_u) 
        
        
        
    new_all_sample_jieci = []
    count_jieci = 0
    if is_count_jieci:
#         print("count_jieci")
        for (s, s_w, m_to_save_item, FC_to_save_list) in all_sample:
            new_FC_to_save_list = []
    #         print("\nls=", FC_to_save_list)
            if FC_to_save_list != []:
                for item in FC_to_save_list:
                    for fc in item[1][:]:
                        if fc.lower() in fileter_jieci:
                            new_FC_to_save_list.append(item)
                            break

            new_all_sample_jieci.append((s, new_FC_to_save_list))

        # count
        for (s, new_FC_to_save_list) in new_all_sample_jieci:
            count_jieci += len(new_FC_to_save_list)
#         print("\ncount_jieci=", count_jieci)
    
    return new_all_sample_u, count_u, new_all_sample_jieci, count_jieci
        
        
        
# 查看恢复数据
def check_recovery(file_path_1, file_path_2):
#     file_path_1, file_path_2
    S___list, S_w_list, m_to_save, FC_to_save, all_sample, length_ = pickle.load(open(file_path_1,"rb"))
#     all_sample_recovery, length_sample, text_recoverability_tosave = pickle.load(open(file_path_2,"rb"))
    all_sample_recovery, length_sample, text_recoverability_tosave = file_path_2
    
    assert len(all_sample_recovery) == len(all_sample)
#     count = 0
    recovery = []
    for (s, _, _, FC_to_save_list), (_, candidate_set_recovery_list) in zip(all_sample, all_sample_recovery):  # (s, candidate_set_recovery) item <--> all_sample_recovery list
#         count+=1
#         print("check_recovery", count)
        if  candidate_set_recovery_list == []:
            continue
        for item in candidate_set_recovery_list:
#             count+=1
#             print("非空", count)
            idx, original_word, candidate_top = item
#             print("item=", item)
            for FC_item in FC_to_save_list:  # (23, ['happiness', 'happy'], 0)
#                 print("FC_item=", FC_item)
                if FC_item[0] == idx:
                    recovery.append((FC_item[1], original_word))
#                     print("\n", FC_item[1], " --> ", original_word)
    
    count_covery_u = 0
    count_covery_jieci = 0
    for item in recovery:
        if item[-1] in fileter_u:
            count_covery_u += 1
#             print("\n", item[0], " --> ", item[-1])
        if item[-1] in fileter_jieci:
            count_covery_jieci += 1
#             print("\n", item[0], " --> ", item[-1])
    return recovery, count_covery_u, count_covery_jieci
        
            
            
# Paraphraser_watermark_files   novels_to_list     novels_to_list_random_m     Paraphraser_watermark_files_random_m    './MLM/novels_to_list/Pride and Prejudice_watermark-500-1500.pk1'
# Dracula    Pride and Prejudice      Wuthering Heights     imdb-100_watermark-no-all-random-seed 
if __name__ == "__main__":
    file_path = './MLM/old_folder/Wuthering Heights_watermark-no-all-random-seed.pk1'
    name = (file_path.split('/')[-1]).split("_")[0]
    recovery = text_recoverability(file_path, (file_path.split('/')[-1]).split("_")[0],  fileter_door=False, t_r=False, )  # 
    
    
#     new_all_sample_u, count_u, new_all_sample_jieci, count_jieci = just_u(file_path)
#     print("\ncount_u=", count_u, "\ncount_jieci=", count_jieci) 
    
    
#     recovery, count_covery_u, count_covery_jieci = check_recovery(file_path, recovery)
#     print("\ncount_covery_u=", count_covery_u, "\ncount_covery_jieci=", count_covery_jieci)
            
            
            






































