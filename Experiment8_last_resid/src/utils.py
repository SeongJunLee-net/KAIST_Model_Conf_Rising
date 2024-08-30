import pandas as pd
import numpy as np
from functools import partial
from copy import deepcopy
from transformers import GPT2LMHeadModel, GPT2Tokenizer, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from datasets import load_dataset


def make_content(examples):
    examples['content'] = ' ' + examples['truncated_input'] + ' ' + examples['output'] 
    return examples

def preprocess_function(examples,tokenizer):
    # batch=True를 통해 하나의 리스트로써 들어온다
    examples = examples['content']
    tokenized = tokenizer(examples, padding=True,truncation=True,return_tensors = 'pt')
    return tokenized

def train(model_type):
    tokenizer = GPT2Tokenizer.from_pretrained(model_type)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'

    train = load_dataset("json", data_files = "./dataset/train.json")
    valid = load_dataset("json", data_files = "./dataset/validation.json")
    preprocess_func = partial(preprocess_function,tokenizer=tokenizer)
    
    train = train.map(make_content)
    valid = valid.map(make_content)


    train_dataset = train.map(preprocess_func,
                      batched = True,
                      num_proc = 4,
                      remove_columns = train['train'].column_names
                      ) 
    
    valid_dataset = valid.map(preprocess_func,
                        batched = True,
                        num_proc = 4,
                        remove_columns = valid['train'].column_names
                        )
    # 왜 저절로 labels가 생기는지는 잘 모르겠음
    # train_dataset = train_dataset.remove_columns(['labels'])
    # valid_dataset =valid_dataset.remove_columns(['labels'])
    
    # 동적 패딩
    data_collator = DataCollatorForLanguageModeling(tokenizer = tokenizer, mlm=False)
    if model_type=='gpt2':
        model = GPT2LMHeadModel.from_pretrained(model_type)
    
    training_args = TrainingArguments(
        output_dir = "./src/data_factory/model_pt/",
        do_train=True,
        do_eval=True,
        num_train_epochs=50,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        seed = 42,
        save_strategy = "epoch",
        eval_strategy = "epoch",
        logging_strategy = "epoch",
        logging_steps = 5,
        learning_rate = 1e-3,
        weight_decay = 0.005,
        save_total_limit=3,           # 저장할 체크포인트의 최대 개수
        load_best_model_at_end=True,  # 최적의 모델을 훈련 종료 시 불러오기
        metric_for_best_model="eval_loss",  # 최적의 모델을 판단할 평가 지표를 'eval_loss'로 지정 GPT2는 CRE
        greater_is_better=False        # 더 높은 지표가 더 좋은 모델임을 의미
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args = training_args,
        train_dataset = train_dataset["train"],
        eval_dataset = valid_dataset["train"],
        data_collator = data_collator
    )

    trainer.train()
    trainer.save_model(output_dir="./src/data_factory/model_pt/best_model")

    return model,tokenizer


def batch(iterable, bsize=1):
    total_len = len(iterable)
    for ndx in range(0, total_len, bsize):
        yield list(iterable[ndx : min(ndx + bsize, total_len)])