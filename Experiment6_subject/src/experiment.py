import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import statistics
import math
import json
from transformers import GPT2LMHeadModel,AutoTokenizer
from tqdm import tqdm
from functools import partial
from utils import batch
from calculate_effect import indirect_calculate,total_calculate,calculate_acc

class Intervention():
    '''
    Wrapper for all the possible interventions
    '''

    def __init__(self,
                 tokenizer,
                 model,
                 label, # label
                 clear_sent, # label context
                 entity_sent, # output context
                 verb,
                 device='cpu'):
        super()
        self.device = device
        self.enc = tokenizer
        self.model = model
        self.enc.pad_token = self.enc.eos_token
        self.enc.padding_side = 'right' # left -> right

        # All the initial strings
        # base_sent, repl_sent
        self.base_strings = [
            clear_sent,
            entity_sent
        ]

        # Tokenized bases
        ''' output 형태
        {'input_ids': tensor([[28541,  1203, 17520,  1371,   373,  9393,   287],
        [50256, 21102,  9520, 21663,   373,  9393,   287]]), 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1],
        [0, 1, 1, 1, 1, 1, 1]])}
        '''
        strings_token = self.enc(self.base_strings,padding = True, return_tensors = 'pt')
        self.base_strings_tok = [

            # clear string(include padding token at left side)
            {
                'input_ids' : strings_token['input_ids'][0].to(device),
                'attention_mask' : strings_token['attention_mask'][0].to(device)
            },

            # entity_string(include padding token at left side)
            {
                'input_ids' : strings_token['input_ids'][1].to(device),
                'attention_mask' : strings_token['attention_mask'][1].to(device)
            }

        ]
        clear_padding_length = (strings_token['attention_mask'][0]==0).sum()
        entity_padding_length = (strings_token['attention_mask'][1]==0).sum()
        
        
        # Where to intervene
        # previous of verb token is representation of subject
        # clear_padding_length +
        # entity_padding_length +
        ## 원래 두 padding_length도 더해줘야함.

        verb_id = self.enc.encode(verb)[0]
        clear_position = strings_token['input_ids'][0].tolist().index(verb_id)-1
        entity_position = strings_token['input_ids'][1].tolist().index(verb_id)-1
        if strings_token['input_ids'][0].tolist().count(verb_id)>1:
            AssertionError
        if strings_token['input_ids'][1].tolist().count(verb_id)>1:
            AssertionError

        # clear_position = clear_sent.split().index(verb)-1
        # entity_position = entity_sent.split().index(verb)-1
        # print(self.enc.decode(strings_token['input_ids'][0][clear_position]))
        # print(self.enc.decode(strings_token['input_ids'][1][entity_position]))
        self.position = (clear_position,entity_position)

        output_id = self.model(**self.base_strings_tok[0])['logits'][-1,:].argmax().item()

        # ,add_space_before_punct_symbol = True
        self.candidate = self.enc.encode(label)
        
        
        # 후보 tokens를 사전의 정의된 숫자로 바꿈
        # output과 label의 token
        self.candidates_tok = [output_id,self.candidate]
                               

class Model():
    def __init__(self,
                 base_model,
                 config,
                 device='cpu',
                 output_attentions=False,
                 masking_approach=1,
                 gpt2_version='gpt2'):
        super()
        # string.startswith 시작 문자열이 지정된 문자열과 같은지 확인(bool)
        self.is_gpt2 = gpt2_version.startswith('gpt2')
        self.device = device
        self.config = config
        self.model = base_model
        self.tokenizer = AutoTokenizer.from_pretrained(
                gpt2_version
            )
        self.order_dims = lambda a: a
        self.model.eval()
        self.model.to(device)
        
        # Options
        self.top_k = 5
        self.num_layers = self.model.config.num_hidden_layers # model의 hidden layer개수를 볼 수 있다. , gpt2,bert의 경우 12
        self.num_neurons = self.model.config.hidden_size # model의 hidden vector size를 볼 수 있다. , gpt2,bert의 경우 768
        self.num_heads = self.model.config.num_attention_heads # model의 attention heads 개수를 볼 수 있다. , gpt2,bert의 경우 12
        

        # Special token id's(st_ids): (mask, cls, sep)
        self.st_ids = (self.tokenizer.mask_token_id,
                       self.tokenizer.cls_token_id,
                       self.tokenizer.sep_token_id)
        if self.is_gpt2:
            self.attention_layer = lambda layer: self.model.transformer.h[layer].attn
            self.word_emb_layer = self.model.transformer.wte
            self.neuron_layer = lambda layer: self.model.transformer.h[layer].mlp

    def get_representations(self, context, position):
        # context = {'input_ids' : , 'attention_mask' : }
        # Hook for saving the representation
        def extract_representation_hook(module,
                                        input,
                                        output,
                                        position, # subject position
                                        representations,
                                        layer):
            
            # 각 layer를 통과할때마다 0번째 token과 {} token의 embedding vector를 representation에 딕셔너리로 저장
            # print("중간 layer output: ",output[self.order_dims((0, position))].size())
            # representation size : [768]
            # output size : [1,5,768]
            representations[layer] = output[self.order_dims((0,position,slice(None)))]

        handles = []
        representation = {}
        with torch.no_grad():
            # construct all the hooks
            # word embeddings will be layer -1
            # 모델의 Word Embedding layer(단어 임베딩, 위치 임베딩, 세그먼트 임베딩중 하나)
            # word embdedding의 forward 사후에 처리할 forward_hook의 handle
            # 이를 partial로 argument를 미리 채운 함수로 만들어 냄으로써 handle을 미리 저장
            # extract_representation_hook의 module = word_emb_layer
                                        
            handles.append(self.word_emb_layer.register_forward_hook(
                partial(extract_representation_hook,
                        position=position,
                        representations=representation,
                        layer=-1)))
            # hidden layers -> gpt2의 경우 12개
            for layer in range(self.num_layers):
                # 각 gpt2 모델의 hidden layer 12개중 하나의 block을 module로 사용
                # 위와 같이 handle로 control하기 위해서 partial로 argument를 미리 채운 함수로 만들어 냄
                # extract_representaion_hook의 module = 각 gpt2 layer block
                handles.append(self.neuron_layer(layer).register_forward_hook(
                    partial(extract_representation_hook,
                            position=position,
                            representations=representation,
                            layer=layer)))
            
            # 각 layer마다의 representation을 {}딕셔너리 형태로 받아오게 된다.
            self.model(**context)
            for h in handles:
                h.remove()

        return representation
    
    def get_probabilities(self,context):
        logits = self.model(**context).logits
        if logits.dim() == 2:
            # print("dim 2 !")
            logits = logits[-1, :] # 다음에 올 단어의 확률로써 단어집합에 확률이 찍힙
        else:
            logits = logits[:,-1, :] # 다음에 올 단어의 확률로써 단어집합에 확률이 찍힙
        probs = F.softmax(logits, dim=-1)
        return probs
    
    def get_prediction(self,context):
        logits = self.model(**context).logits
        if logits.dim() == 2:
            # print("dim 2 !")
            logits = logits[-1, :] # 다음에 올 단어의 확률로써 단어집합에 확률이 찍힙
        else:
            logits = logits[:,-1, :] # 다음에 올 단어의 확률로써 단어집합에 확률이 찍힙
        probs = F.softmax(logits, dim=-1)
        return probs.argmax()


    def get_probabilities_for_examples(self, context, candidates):
        """Return probabilities of single-token candidates given context"""
        for c in candidates:
            if len(c) > 1:
                raise ValueError(f"Multiple tokens not allowed: {c}")
        # print("candidates: ",candidates)
        outputs = [c[0] for c in candidates] # label과 output의 ids
        logits = self.model(**context).logits #logit을 내보냄
        # print(logits.size())
        # sequence_length, vocab_size
        if logits.dim() == 2:
            # print("dim 2 !")
            logits = logits[-1, :] # 다음에 올 단어의 확률로써 단어집합에 확률이 찍힙
        else:
            logits = logits[:,-1, :] # 다음에 올 단어의 확률로써 단어집합에 확률이 찍힙
            outputs = self.order_dims((slice(None),outputs))
            # print(outputs)
        probs = F.softmax(logits, dim=-1) # 확률처리
        return probs[outputs].tolist() 

    def get_probabilities_for_examples_multitoken(self, context, candidates):
        """
        Args:
            context: Tensor of token ids in context
            -> 앞서 등장할 문장 {'input_ids': ,'attention_mask': }
            candidates: list of list of token ids in each candidate
            -> 이후에 올 문장에 대한 token ids의 리스트의 리스트
        
        Returns: list containing probability for each candidate
            -> 각 candidate에 대한 확률 리스트
        """
        # TODO: Combine into single batch
        mean_probs = []
        context = context['input_ids'].tolist()
        # print("context: ",context)
        for candidate in candidates: # candidate별로 수행
            token_log_probs = []
            combined = context + candidate
            # Exclude last token position when predicting next token
            # 마지막 토큰을 제외함으로써, 현재 후보인 토큰에 대한 확률만 계산한다.
            batch = torch.tensor(combined[:-1]).unsqueeze(dim=0).to(self.device)
            # Shape (batch_size, seq_len, vocab_size)
            # 여기선 batch_size = 1
            logits = self.model(batch)[0]
            # Shape (seq_len, vocab_size)\
            # print(logits.size())
            log_probs = F.log_softmax(logits[-1, :, :], dim=-1)

            context_end_pos = len(context) - 1
            continuation_end_pos = context_end_pos + len(candidate)
            # TODO: Vectorize this
            # Up to but not including last token position
            for i in range(context_end_pos, continuation_end_pos):
                next_token_id = combined[i+1]
                next_token_log_prob = log_probs[i][next_token_id].item()
                token_log_probs.append(next_token_log_prob)
        # 이후에 나올 확률을 종합(평균)한다.
        # log를 씌웠으니 다시 exponential 취해줌
        # 이런식으로 각 candidate tokens가 나올 확률을 종합
            mean_token_log_prob = statistics.mean(token_log_probs)
            mean_token_prob = math.exp(mean_token_log_prob)
            mean_probs.append(mean_token_prob)
        return mean_probs
                    
    def neuron_intervention(self,
                            context,# base sentence dict {'input_ids': , 'attention_mask': }
                            outputs,# [[label_ids],[output_ids]]
                            rep,# 각 layer 별 representation(key = -1,0,1,...,11)[clear]
                            layers,# search할 neuron및 layer 리스트(e.g. layers = [0],[1],..)
                            neurons,# e.g. layer 1의 neuron 0:767
                            # position = (base_position, repl_position)
                            position,
                            # Indirect = replace
                            intervention_type='replace',
                            alpha=1.):
        
        # Hook for changing representation during forward pass
        def intervention_hook(module,
                              # (batch_size, seq_len)
                              input, 
                              # (batch_size, seq_len, # of neurons)
                              output, 
                              # subject의 마지막 토큰의 위치(clear, entity)
                              position,
                              # neuron의 정렬 리스트 [[0],[1],[2],..[767]] 
                              neurons, 
                              # rep의 해당하는 neuron에 alpha를 곱함
                              # clear
                              intervention, 
                              # replace or diff
                              intervention_type):
            # Get the neurons to intervene on
            neurons = torch.LongTensor(neurons).to(self.device)
            # First grab the position across batch
            # Then, for each element, get correct index w/ gather
            # slice(None)은 : 와 동일하게 동작
            entity_position = position[1]
            # print("base_position: ",base_position)
            entity_slice = self.order_dims((slice(None), entity_position, slice(None)))

            # take a representation on the base_position
            # and gather them where position is neurons.
            base = output[entity_slice].gather(1, neurons)

            # intervention(clear의 layer별 neuron)을 base의 크기로 바꿈
            intervention_view = intervention.view_as(base)

            if intervention_type == 'replace':
                # base로 replace를 진행
                base = intervention_view
            else:
                raise ValueError(f"Invalid intervention_type: {intervention_type}")
            
            # Overwrite values in the output
            # First define mask where to overwrite
            scatter_mask = torch.zeros_like(output, dtype=torch.bool)
            for i, v in enumerate(neurons):
                # 각 batch마다 다른 neuron에 1을 부여
                scatter_mask[self.order_dims((i, entity_position, v))] = 1

            # masked_scatter_ : scatter_mask가 true인 위치에 base.flatten()의 값을 순서대로 넣음.
            output.masked_scatter_(scatter_mask, base.flatten())

        # Set up the context as batch
        batch_size = len(neurons) # 768

        # batch_size 만큼 반복함으로써 각 neuron을 intervention했을 때 결과를 봄
        input_ids = context['input_ids'].unsqueeze(0).repeat(batch_size, 1)
        attention_mask = context['attention_mask'].unsqueeze(0).repeat(batch_size, 1)

        context = { # corrupted input 
            'input_ids' : input_ids,
            'attention_mask' : attention_mask
        }

        handle_list = []

        # search할 layer를 대상으로
        # neurons = [[0],[1],...]
        # print(layers)
        # print(neurons)
        for layer in set(layers):
            neuron_loc = np.where(np.array(layers) == layer)[0]
            n_list = []
            for n in neurons:
                unsorted_n_list = [n[i] for i in neuron_loc]
                n_list.append(list(np.sort(unsorted_n_list)))
            
            # layer의 각 neuron에 해당하는 부분에 alpha=1를 곱함
            # (number of neurons, 1)
            intervention_rep = alpha * rep[layer][n_list]
            
            # embedding layer와 neuron layer가 다르기 때문에 나눔
            if layer == -1:
                # forward hook을 거는데, module은 word_emb_layer 
                # 이 forward hook을 각 layer에 대한 output에 걸음으로써
                # replace하는 효과를 낸다. 
                handle_list.append(self.word_emb_layer.register_forward_hook(
                    partial(intervention_hook,
                            # subject의 마지막 토큰위치
                            position=position, 
                            #[[0],[1],[2],...[767]]
                            neurons=n_list, 
                            intervention=intervention_rep,
                            # Indirect = replace
                            intervention_type=intervention_type)))
            else:
                handle_list.append(self.neuron_layer(layer).register_forward_hook(
                    partial(intervention_hook,
                            position=position,
                            neurons=n_list,
                            intervention=intervention_rep,
                            intervention_type=intervention_type)))
        # print("context: ",context)
        # print("output: ",outputs)
        new_probabilities = self.get_probabilities(context)
        for hndle in handle_list:
            hndle.remove()
        # 도출된 확률리스트를 return
        return new_probabilities
    
    def effect_index(self):
        if self.config['trained'] == True:
            te_data = pd.read_csv(self.config['data_dir']+f"gpt2_train=True_Total.csv")
            if self.config['editing']=='prob':
                positive_effect = te_data.loc[te_data['prob']>0].index
                negative_effect = te_data.loc[te_data['prob']<0].index
            elif self.config['editing']=='entropy':
                positive_effect = te_data.loc[te_data['entropy']>0].index
                negative_effect = te_data.loc[te_data['entropy']<0].index
        else:
            te_data = pd.read_csv(self.config['data_dir']+f"gpt2_train=False_Total.csv")
            if self.config['editing']=='prob':
                positive_effect = te_data.loc[te_data['prob']>0].index
                negative_effect = te_data.loc[te_data['prob']<0].index
            elif self.config['editing']=='entropy':
                positive_effect = te_data.loc[te_data['entropy']>0].index
                negative_effect = te_data.loc[te_data['entropy']<0].index
        return (positive_effect,negative_effect)
    
    def edit_representation(self,edit_data,intervention,layers,method):
        # edit_data size = (len(samples), neuron_size, layer size)
        def edit_hook(module,
                      input,
                      output,
                      test_rep,
                      position,
                      # (num of neurons,)
                      rep): 
            clear_position = position
            clear_slice = self.order_dims((0,clear_position,slice(None)))
            ## 실제로 찍어보면 같은 값을 냄
            # print(output[clear_slice])
            # print(test_rep)
            output[clear_slice] = output[clear_slice] + rep
            ## seed가 바뀌어서 다르다고 나오는듯
            # if torch.equal(output[clear_slice],test_rep):
            #     output[clear_slice] = output[clear_slice] + rep
            # else:
            #     print("output_rep: ", output[clear_slice].size())
            #     print("test_rep: ",test_rep.size())

        handle_list = []
        prob_list = []
        individual_prob_list = []
        pred_list = []
        indiv_pred_list = []
        edit_data = edit_data.to(self.device)

        clear_representations = self.get_representations(
            intervention.base_strings_tok[0], # clear sent e.g. {'input_ids': ~~,'attention_mask':~~}
            intervention.position[0]) # clear sent position e.g. 1
        corrupted_representations = self.get_representations(
            intervention.base_strings_tok[1],
            intervention.position[1])
        
        # make mask
        total_mask = torch.zeros_like(edit_data,dtype=bool)
        for layer in range(-1,self.num_layers):
            clear_pt = clear_representations[layer]
            cor_pt = corrupted_representations[layer]
            mask1 = torch.where(clear_pt<cor_pt,True,False)
            total_mask[:,layer+1] = mask1
        # 방향 조절
        edit_data = torch.where(total_mask, -1*edit_data, edit_data)
    
        context = intervention.base_strings_tok[0] # clear context
        position = intervention.position[0] # clear position
        #output_id = intervention.candidates_tok[0]

        origin_prob = method(self.get_probabilities(context))
        prob_list.append(origin_prob)
        individual_prob_list.append(origin_prob)
        # origin prediction
        origin_pred = self.get_prediction(context).item()
        pred_list.append(origin_pred)
        indiv_pred_list.append(origin_pred)
        
        for layer in layers:
            hook_function = partial(edit_hook,
                            test_rep=clear_representations[layer],
                            position = position,
                            rep = edit_data[:,layer+1])
            if layer==-1:
                handle_list.append(self.word_emb_layer.register_forward_hook(hook_function))
            else:
                handle_list.append(self.neuron_layer(layer).register_forward_hook(hook_function))

            # These two line can be replaced efficiently.
            prob_list.append(method(self.get_probabilities(context))) # layer -1 부터 전체 layer에 대해서 Edit
            new_pred = self.get_prediction(context).item()
            if type(new_pred)!=int:
                raise AssertionError
            pred_list.append(new_pred)
        
        for hndle in handle_list:
            hndle.remove()
        
        for layer in layers:
            hook_function = partial(edit_hook,
                            test_rep=clear_representations[layer],
                            position = position,
                            rep = edit_data[:,layer+1])
            if layer==-1:
                handle_list.append(self.word_emb_layer.register_forward_hook(hook_function))
            else:
                handle_list.append(self.neuron_layer(layer).register_forward_hook(hook_function))

            individual_prob_list.append(method(self.get_probabilities(context))) # layer -1 부터 전체 layer에 대해서 Edit

            new_pred = self.get_prediction(context).item()
            if type(new_pred)!=int:
                raise AssertionError
            indiv_pred_list.append(new_pred)

            for hndle in handle_list:
                hndle.remove()

        return prob_list,individual_prob_list,pred_list,indiv_pred_list
                        
    
    def layer_edit_experiment(self,ie_data,interventions,method):
        if type(self.model)==GPT2LMHeadModel:
            layer = [f'layer {i}' for i in range(13)]
            layer.insert(0,'origin')
        else: AssertionError
        # 정답 범위를 좁히게 되면 len(self.tokenizer줄이기)
        layers = list(range(-1,self.num_layers))
        return_data = pd.DataFrame(columns = layer)
        indiv_return_data = pd.DataFrame(columns = layer)

        acc_target_lst = []
        acc_pred_lst = []
        acc_indiv_pred_lst = []
        
        for idx,(_,v) in tqdm(enumerate(ie_data.items())):
            # size = (len(neuron),len(layer))
            v_pt = torch.FloatTensor(v)
            norm = v_pt.norm(dim=0).unsqueeze(0)
            # 이때 뒤 layer로 갈수록 전체 Sum이 커짐(Analysis2.ipynb 참고)
            v_pt = v_pt/norm
    
            intervention = interventions[idx]
            
            tmp,indiv_tmp,pred_lst,indiv_pred_lst = \
                self.edit_representation(v_pt,intervention,layers,method)
            
            acc_target_lst.append(intervention.candidate[0])
            acc_pred_lst.append(pred_lst)
            acc_indiv_pred_lst.append(indiv_pred_lst)

            tmp = pd.DataFrame([tmp], columns = layer)
            indiv_tmp = pd.DataFrame([indiv_tmp], columns = layer)

            return_data = pd.concat([return_data,tmp],ignore_index=True)
            indiv_return_data = pd.concat([indiv_return_data,indiv_tmp],ignore_index=True)


        return_data.reset_index(inplace=True,drop=True)
        indiv_return_data.reset_index(inplace=True,drop=True)

        te_index = self.effect_index() # return positive index, negative index
        acc_df = calculate_acc(acc_target_lst,acc_pred_lst,acc_indiv_pred_lst,te_index)
        
        return return_data,indiv_return_data,acc_df
        

    def neuron_intervention_experiment(self,
                                       id2intervention, # {0: Intervention객체1, 1:Intervention객체2,...}
                                       intervention_type, # ['Indirect','Total']
                                       layers_to_adj=[],
                                       neurons_to_adj=[],
                                       alpha=1,
                                       intervention_loc='all'):


        id2intervention_proposed = {}
        id2intervention_prob = {}
        id2intervention_kl = {}
        id2intervention_entropy = {}

        total_effect = {#'proposed':[],
                        'prob':[],
                        'kl':[],
                        'entropy':[]}
        if intervention_type=='Indirect':
            for id in tqdm(id2intervention, desc='id'):
                # dataframe return 
                id2intervention_prob[id], id2intervention_kl[id], id2intervention_entropy[id] \
                    = self.neuron_intervention_single_experiment(
                    id2intervention[id],
                    intervention_type, 
                    layers_to_adj, neurons_to_adj,
                    alpha,
                    intervention_loc=intervention_loc)
            return id2intervention_prob,id2intervention_kl,id2intervention_entropy
                
        elif intervention_type=='Total':
            for id in tqdm(id2intervention, desc='id'):
                total_results\
                    = self.neuron_intervention_single_experiment(
                    id2intervention[id],
                    intervention_type, 
                    layers_to_adj, neurons_to_adj,
                    alpha,
                    intervention_loc=intervention_loc)
                #total_effect['proposed'].append(total_results[0])
                total_effect['prob'].append(total_results[0])
                total_effect['kl'].append(total_results[1])
                total_effect['entropy'].append(total_results[2])

            return total_effect

        else:
            raise AssertionError
        

    def neuron_intervention_single_experiment(self,
                                              intervention, # label에 대한 Intervention 객체
                                              intervention_type,
                                              layers_to_adj=[],
                                              neurons_to_adj=[],
                                              alpha=100,
                                              bsize=800, intervention_loc='all'):
        """
        run one full neuron intervention experiment
        """
        with torch.no_grad():
            '''
            Compute representations for base terms
            '''
            # 각 layer별 768차원의 representation값
            clear_representations = self.get_representations(
                intervention.base_strings_tok[0], # clear sent e.g. {'input_ids': ~~,'attention_mask':~~}
                intervention.position[0]) # clear sent position e.g. 1
            
            entity_representations = self.get_representations(
                intervention.base_strings_tok[1], # entity sent e.g. {'input_ids': ~~,'attention_mask':~~}
                intervention.position[1]) # e.g. 0
            
            # 50257(vocab.size)개수의 word probability
            clear_prob= self.get_probabilities(intervention.base_strings_tok[0])
            # max_idx = clear_prob.argmax()
            # print(self.tokenizer.decode([max_idx]))
            # print(torch.topk(clear_prob,10).indices)
            corrupted_prob = self.get_probabilities(intervention.base_strings_tok[1])
            # max_idx = corrupted_prob.argmax()
            # print(self.tokenizer.decode([max_idx]))
            # print(torch.topk(corrupted_prob,10).indices)
            
            if intervention_type =='Total':
                prob,kl,entropy = \
                    total_calculate(clear_prob,corrupted_prob,intervention.candidates_tok)
                
                return prob,kl,entropy

            
            # Now intervening on hallucination example
            # self.num_layers = 12, self.num_neurons = 768
            elif intervention_type == 'Indirect':
                context = intervention.base_strings_tok[1] # entity input
                rep = clear_representations # clear input representation
                patched_prob = torch.zeros((self.num_layers + 1, self.num_neurons, len(self.tokenizer.vocab)))
                for layer in range(-1, self.num_layers):
                    # batch 형태의 list로 나눔
                    # 각 layer의 neuron을 batch 형태로 넣고 search한다.
                    # bsize = batch size = 800
                    for neurons in batch(range(self.num_neurons), bsize):
                        # print(neurons) # 0~767 list

                        # neurons_to_adj default = [], layers_to_adj default = []
                        ## neurons_to_search = [[0],[1],[2],[3],..]
                        ## layers_to_search = [-1]
                        neurons_to_search = [[i] + neurons_to_adj for i in neurons]
                        layers_to_search = [layer] + layers_to_adj
                        
                        # replace 이후의 label or output의 prob. (size : [768,50257])
                        patched_layer = self.neuron_intervention(
                            context=context, # base sentence dict {'input_ids': , 'attention_mask': }
                            outputs=intervention.candidates_tok, # [[label_ids],[output_ids]]
                            rep=rep, # 각 layer 별 representation(key = -1,0,1,...,11)

                            # search할 neuron및 layer 리스트
                            # e.g. layer 1의 neuron 0:767
                            layers=layers_to_search, 
                            neurons=neurons_to_search, 

                            # padding포함 verb 이전위치
                            # position = (clear subject position, entity position)
                            position=intervention.position, 

                            # Indirect = replace
                            intervention_type='replace', 
                            alpha=alpha)
                        
                        # 도출된 확률을 저장
                        # embedding layer = -1 ...
                        patched_prob[layer + 1, :, :] = patched_layer
        
                assert ((clear_prob>1) & (clear_prob<0)).sum() == 0
                assert ((corrupted_prob>1) & (corrupted_prob<0)).sum() == 0
                assert ((patched_prob>1) & (patched_prob<0)).sum() == 0
                                
                prob,kl,entropy = indirect_calculate(clear_prob,corrupted_prob,patched_prob,intervention.candidates_tok)
                

                return prob, kl, entropy
            
            else:
                raise SyntaxError
