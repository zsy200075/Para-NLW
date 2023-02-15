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
from nltk.tokenize import sent_tokenize, word_tokenize  

from bert_gen import bert_gen   


# Paraphraser_watermark_files   novels_to_list           './MLM/novels_to_list/Pride and Prejudice_watermark-500-1500.pk1'
# Dracula    Pride and Prejudice
def count_candidate_set(all_sample):
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

bert_tokenizer = BertTokenizer.from_pretrained("bert-base-cased", cache_dir='./pretrained_models/bert-base-cased',)   

src_vocab_file = "./pretrained_models/bert-base-cased/6508e60ab3c1200bffa26c95f4b58ac6b6d95fba4db1f195f632fa3cd7bc64cc.437aa611e89f6fc6675a049d2b5545390adbc617e7d655286421c191d2be2791"
with open(src_vocab_file, 'r', encoding='utf-8') as f:
    lines = f.read().split('\n')
inverse_src_vocab = dict(zip(src_vocab.values(),src_vocab.keys()))




def read_file_get_all_sample(file_path):
    S___list, S_w_list, m_to_save, FC_to_save, all_sample, length_ = pickle.load(open(file_path,"rb"))
    return all_sample


def text_recoverability(file_path, name_, t_r=True, fileter_door=True):
    file_path = file_path
    tag = '_'.join((file_path.split('/')[2]).split('_')[:2])
    name = ((file_path.split('/')[-1]).split(".")[0]).split('_')[0] + (((file_path.split('/')[-1]).split(".")[0]).split('_')[1]).split('mark')[-1]
 
    S___list, S_w_list, m_to_save, FC_to_save, all_sample, length_ = pickle.load(open(file_path,"rb"))
    novel_to_list, novel_to_list_length = pickle.load(open("./MLM/novels_to_list/{}.pk1".format(name_),"rb"))
    num_samples = len(all_sample)
     
    count_ls, count_kong, count_one, count_two, count_three, count_four = count_candidate_set(all_sample)
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

            if tag == 'old_folder':
                sentence_word = bert_tokenizer.tokenize(sentence)  
                sentence_w_word = bert_tokenizer.tokenize(sentence_w)

            if tag == 'new_folder':
                sentence_word = word_tokenize(sentence)  
                sentence_w_word = word_tokenize(sentence_w) 

            candidate_set_recovery = []
            for item in FC_to_save_item:
                (idx, candidate_set, _) = item
                original_word = sentence_word[idx-1]
                copy_sentence_w_word = copy.deepcopy(sentence_w_word)
                   candidate_A, candidate_B = candidate_set

                copy_sentence_w_word[idx-1] = '[MASK]'
                copy_sentence_w = " ".join(copy_sentence_w_word)
                input_tokenizer = bert_tokenizer(copy_sentence_w, padding=True, truncation=True, return_tensors='pt')
                input_2id = (input_tokenizer["input_ids"]).cuda()
                attention_mask = (input_tokenizer["attention_mask"]).cuda()  
   
                mask_ = (input_2id == bert_tokenizer.mask_token_id).cpu()
                mask_ = mask_.numpy()
                mask_ = mask_.astype(int) 
                mask_ = torch.from_numpy(mask_)
                mask_.cuda()

                with torch.no_grad():
                    output = bert_gen(input_2id, mask_, attention_mask)
                output = output.view(-1)
                output_list = output.detach().cpu().numpy()
                output_dict_id_2word = dict(zip(list(range( bert_tokenizer.vocab_size)), list(output_list)))
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
    
                candidate_set_recovery.append((idx, original_word, candidate_top))

            for item in candidate_set_recovery:

                if item[1] == item[2]:
                    count_recovery += 1 

            all_sample_recovery.append((s, candidate_set_recovery)) 

           print("count_recovery=", count_recovery, "\n") 

        text_recoverability =  count_recovery/count_ls
        text_recoverability_tosave = 'percent: {:.2%}'.format(text_recoverability)
        print(text_recoverability_tosave) 
        if not os.path.exists("./MLM/text_recoverability"):
            os.makedirs("./MLM/text_recoverability", exist_ok=True)
        name_ = tag + "_" + name 
        print("{}写入结束".format(name_))
        
        return (all_sample_recovery, len(all_sample_recovery), text_recoverability_tosave)
        
if __name__ == "__main__":
    file_path = './MLM/old_folder/Wuthering Heights_watermark-no-all-random-seed.pk1'
    name = (file_path.split('/')[-1]).split("_")[0]
    recovery = text_recoverability(file_path, (file_path.split('/')[-1]).split("_")[0],  fileter_door=False, t_r=False, )  # 
    
  





































