import os
import pandas as pd
import json
import random
import numpy as np
import argparse
import yaml
import torch
from calculate_effect import entropy,prob_diff
from experiment import Intervention, Model
from transformers import GPT2Tokenizer,GPT2LMHeadModel
from datetime import datetime
from utils import train

os.chdir("/home/sjlee/hdd1/NLP/Experiment8")

'''
implementation tactic:
following the original code. 
if i don't know what is it then do annotation.
Model : GPT2
'''

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Many Change
def construct_interventions(data,tokenizer,base_model,DEVICE,config):
    interventions = {}
    with open('./dataset/verb.json','r') as f:
        rel2verb = json.load(f)
    for cnt,item in data.iterrows():
        label = ' ' + item['output'] # label
        clear_sent = ' ' + item['truncated_input']
        entity_sent = ' ' + item['entity_input']
        verb = rel2verb[item['rel_id']]

        # 원래는 target인데 target이 중복이 존재해 수정
        interventions[cnt] = Intervention(
            tokenizer,
            base_model,
            label,
            clear_sent,
            entity_sent,
            verb,
            token_pos = config['token_pos'],
            device = DEVICE
        )
    
    return interventions

def read_config(config_path):
    print(os.getcwd())
    with open(config_path,'rb') as f:
        config = yaml.load(f,Loader=yaml.FullLoader)
    
    
    return config


def run_all():
    seed_everything(42)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    config = read_config('./config/gpt2.yaml')

    model_type = config['model']
    intervnetion_types = config['intervention_type']
    editing = config['editing']

    data_name = config['data_name']
    data_dir = config['data_dir']
    out_dir = config['out_dir']

    test_data = pd.read_csv(data_dir+"./test.csv")

    print("Model:", model_type)
    
    if config['trained'] == True:
        model_path = './src/data_factory/model_pt/best_model'
        if not os.path.exists(model_path):
            base_model,tokenizer = train(model_type)
        else:
            base_model = GPT2LMHeadModel.from_pretrained(model_path)
            tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    else:
        if model_type=='gpt2':
            base_model = GPT2LMHeadModel.from_pretrained(model_type)
            tokenizer = GPT2Tokenizer.from_pretrained(model_type)

        
    # set up all output
    # Initialize Model and Tokenizer.
    model = Model(base_model=base_model,device=device,config =config)
        
    # Set up folder if it does not exist.
    dt_string = datetime.now().strftime("%Y%m%d")
    folder_name = dt_string + "_neuron_intervention"
    base_path = os.path.join(out_dir, "results", folder_name)


    if not os.path.exists(base_path):
        os.makedirs(base_path)

    if not os.path.exists(os.path.join(base_path,data_name)):
        os.makedirs(os.path.join(base_path,data_name))
    

    interventions = construct_interventions(test_data,tokenizer,base_model,device,config)
    
    
    if editing=='entropy':    
        with open(data_dir+f"gpt2_train={config['trained']}_entropy.json",'rb') as f:
            ie_data = json.load(f)
            method = entropy
        edit_results, edit_results_indiv, acc_df = model.layer_edit_experiment(ie_data,interventions,method)

        edit_results.to_csv(data_dir+f"gpt2_train={config['trained']}_edit_{editing}.csv",index=False)
        edit_results_indiv.to_csv(data_dir+f"gpt2_train={config['trained']}_edit_indiv_{editing}.csv",index=False)
        acc_df.to_csv(data_dir+f"gpt2_train={config['trained']}_edit_accuracy__{editing}.csv",index=False)
        return 1
    
    if editing=='prob':
        with open(data_dir+f"gpt2_train={config['trained']}_prob.json",'rb') as f:
            ie_data = json.load(f)
            method = prob_diff
        edit_results, edit_results_indiv, acc_df = model.layer_edit_experiment(ie_data,interventions,method)

        edit_results.to_csv(data_dir+f"gpt2_train={config['trained']}_edit_{editing}.csv",index=False)
        edit_results_indiv.to_csv(data_dir+f"gpt2_train={config['trained']}_edit_indiv_{editing}.csv",index=False)
        acc_df.to_csv(data_dir+f"gpt2_train={config['trained']}_edit_accuracy__{editing}.csv",index=False)
        return 1
    
    if intervnetion_types==False:
        print('end')
        return 1
    
    for itype in intervnetion_types:
        # proposed, prob, kl, entropy
        intervention_results = model.neuron_intervention_experiment(
            interventions,itype,alpha = 1.0
        )

        if itype == 'Indirect':
            ## 보류
            # with open(data_dir + f"{model_type}_train={config['trained']}_proposed.json",'w') as f:
            #     json.dump(intervention_results[0],f)
            with open(data_dir + f"{model_type}_train={config['trained']}_prob.json",'w') as f:
                json.dump(intervention_results[0],f)
            with open(data_dir + f"{model_type}_train={config['trained']}_kl.json",'w') as f:
                json.dump(intervention_results[1],f)
            with open(data_dir + f"{model_type}_train={config['trained']}_entropy.json",'w') as f:
                json.dump(intervention_results[2],f)
        elif itype == 'Total':
            # 'interventions' type is dictionary
            save_file = pd.DataFrame(intervention_results)
            save_file.to_csv(data_dir+f"{model_type}_train={config['trained']}_Total.csv",index=False)            
        


if __name__ == "__main__":
    run_all()
