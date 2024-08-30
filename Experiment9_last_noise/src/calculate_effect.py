import os
import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from collections import defaultdict
from sklearn.metrics import accuracy_score
from tqdm import tqdm

def load_config(config_path):
    with open(config_path,'rb') as f:
        config = yaml.load(f,Loader = yaml.FullLoader)
    return config
def total_calculate(clear_prob, corrupted_prob, candidates_token):
    # patched_prob : (num of layer, num of neurons, len(vocab))
    # candidates_token = (output_id, label_id)
    device = "cuda:0" if torch.cuda.is_available() else ("cpu")
    clear_prob_pt = clear_prob.detach().to(device)
    corrupted_prob_pt = corrupted_prob.detach().to(device)
    
    # bool_mask = torch.zeros_like(clear_prob_pt,dtype=bool).to(device) 
    # bool_mask[candidates_token[0]] = True

    # proposed
    # prob_diff = corrupted_prob_pt-clear_prob_pt
    # prob_diff = torch.where(bool_mask, -1 * prob_diff, prob_diff)
    # proposed = prob_diff.sum().to("cpu") # summation along the vocab. set

    # prob
    clear_output_prob = clear_prob_pt.max()
    corrupted_output_prob = corrupted_prob_pt.max()
    prob = (clear_output_prob-corrupted_output_prob).to("cpu") 

    # kl (using natural log)
    log_clear_prob_pt = clear_prob_pt.log()
    log_corrupted_prob_pt = corrupted_prob_pt.log()

    first = F.kl_div(input = log_corrupted_prob_pt,target = clear_prob_pt,reduction='none').sum() # row = # of neurons
    second = F.kl_div(input = log_clear_prob_pt,target = clear_prob_pt,reduction='none').sum()
    kl = (first-second).to("cpu")

    # entropy     
    clear_entropy = (-1*clear_prob_pt*log_clear_prob_pt).sum()
    corrupted_entropy = (-1*corrupted_prob_pt*log_corrupted_prob_pt).sum()
    
    # entropy가 얼마나 작아졌는지 확인
    entropy = (corrupted_entropy-clear_entropy).to("cpu")

    # tensor(x.xx)에 tolist() 사용시 float로 변환
    return prob.tolist(),kl.tolist(),entropy.tolist()


def indirect_calculate(clear_prob, corrupted_prob, patched_prob, candidates_token):
    # patched_prob : (num of layer, num of neurons, len(vocab))
    # candidates_token = (output_id, label_id)
    device = "cuda:0" if torch.cuda.is_available() else ("cpu")
    clear_prob_pt = clear_prob.detach().to(device).unsqueeze(0).unsqueeze(0).repeat(patched_prob.size(0),patched_prob.size(1),1)
    corrupted_prob_pt = corrupted_prob.detach().to(device).unsqueeze(0).unsqueeze(0).repeat(patched_prob.size(0),patched_prob.size(1),1)
    patched_prob_pt = patched_prob.detach().to(device)

    # clear_prob = clear_prob.detach()
    # corrupted_prob = corrupted_prob.detach()
    # patched_prob = patched_prob.detach()

    # bool_mask = torch.zeros_like(patched_prob,dtype=bool).to(device) 
    # bool_mask[:,:,candidates_token[0]] = True
    
    width = patched_prob.size(1)
    height = patched_prob.size(0)
    #proposed = torch.zeros((height,width))
    prob = torch.zeros((height,width))
    kl = torch.zeros((height,width))
    entropy = torch.zeros((height,width))

    # proposed
    # prob_diff = corrupted_prob_pt-patched_prob_pt
    # prob_diff = torch.where(bool_mask, -1 * prob_diff, prob_diff)
    # proposed = prob_diff.sum(axis=2).to("cpu") # summation along the vocab. set

    # prob
    patched_output_prob = patched_prob_pt.max(axis=2).values
    corrupted_output_prob = corrupted_prob_pt.max(axis=2).values
    prob = (patched_output_prob-corrupted_output_prob).to("cpu") 

    # kl (using natural log)
    log_patched_prob_pt = patched_prob_pt.log()
    log_corrupted_prob_pt = corrupted_prob_pt.log()

    first = F.kl_div(input = log_corrupted_prob_pt,target = clear_prob_pt,reduction='none').sum(axis=2) # row = # of neurons
    second = F.kl_div(input = log_patched_prob_pt,target = clear_prob_pt,reduction='none').sum(axis=2)
    kl = (first-second).to("cpu")

    # entropy     
    patched_entropy = (-1*patched_prob_pt*log_patched_prob_pt).sum(axis=2)
    corrupted_entropy = (-1*corrupted_prob_pt*log_corrupted_prob_pt).sum(axis=2)
    
    # entropy가 얼마나 작아졌는지 확인
    entropy = (corrupted_entropy-patched_entropy).to("cpu")


    return prob.T.tolist(),kl.T.tolist(),entropy.T.tolist()

def entropy(prob):
    log_prob = prob.log()
    return_prob = (-prob*log_prob).sum()
    return return_prob.item()

def prob_diff(prob):
    return_prob = prob.max().item()
    if type(return_prob) != float:
        print(type(return_prob))
        raise AssertionError
    return return_prob

def calculate_acc(acc_target_lst,acc_pred_lst,acc_indiv_pred_lst,te_index:tuple):
    # acc_target_lst = 1 dimensional list
    # acc_pred_lst = 2 dimensional list (e.g. [[origin, edit layer 0, edit layer 0&1 ..],[],[]] )
    # acc_indiv_pred_lst = 2 dimensional list (e.g. [[origin, edit layer 0, edit layer 1 ..],[],[]] )

    layers = [f'layer {i}' for i in range(13)]
    layers.insert(0,'origin')
    
    # shape = (# of samples)
    acc_target_arr = np.array(acc_target_lst)
    # shape = (# of samples, # of layers+1)
    acc_pred_arr = np.array(acc_pred_lst)
    # shape = (# of samples, # of layers+1)
    acc_indiv_pred_arr = np.array(acc_indiv_pred_lst)

    return_dict = defaultdict(list)

    for cnt,layer in enumerate(layers):
        pred = acc_pred_arr[:,cnt]
        acc_score = accuracy_score(y_true = acc_target_arr,y_pred = pred)
        return_dict['total_stacked'].append(acc_score)

        indiv_pred = acc_indiv_pred_arr[:,cnt]
        acc_score_indiv = accuracy_score(y_true=acc_target_arr,y_pred = indiv_pred)
        return_dict['total_indiv'].append(acc_score_indiv)

        pos_target = acc_target_arr[te_index[0]]
        pos_pred = acc_pred_arr[te_index[0],cnt]
        pos_pred_indiv = acc_indiv_pred_arr[te_index[0],cnt]
        pos_acc_score = accuracy_score(y_true = pos_target,y_pred = pos_pred)
        pos_acc_score_indiv = accuracy_score(y_true = pos_target,y_pred = pos_pred_indiv)
        return_dict['pos_stacked'].append(pos_acc_score)
        return_dict['pos_indiv'].append(pos_acc_score_indiv)

        neg_target = acc_target_arr[te_index[1]]
        neg_pred = acc_pred_arr[te_index[1],cnt]
        neg_pred_indiv = acc_indiv_pred_arr[te_index[1],cnt]
        neg_acc_score = accuracy_score(y_true = neg_target,y_pred = neg_pred)
        neg_acc_score_indiv = accuracy_score(y_true = neg_target,y_pred = neg_pred_indiv)
        return_dict['neg_stacked'].append(neg_acc_score)
        return_dict['neg_indiv'].append(neg_acc_score_indiv)

    return_df = pd.DataFrame(return_dict,index=layers)

    return_df = return_df[['total_stacked','pos_stacked','neg_stacked','total_indiv','pos_indiv','neg_indiv']]
    
    return return_df


        



    


