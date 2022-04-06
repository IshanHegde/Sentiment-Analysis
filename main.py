#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 16:24:07 2022

@author: ishanhegde
"""

import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
import matplotlib.pyplot as plt
# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import logging
from scipy.spatial.distance import cosine
from nltk import sent_tokenize
import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

def load_data(split_name='train', columns=['text', 'stars'], folder='data'):
    '''
        "split_name" may be set as 'train', 'valid' or 'test' to load the corresponding dataset.
        
        You may also specify the column names to load any columns in the .csv data file.
        Among many, "text" can be used as model input, and "stars" column is the labels (sentiment). 
        If you like, you are free to use columns other than "text" for prediction.
    '''
    try:
        print(f"select [{', '.join(columns)}] columns from the {split_name} split")
        df = pd.read_csv(f'{folder}/{split_name}.csv')
        df = df.loc[:,columns]
        print("Success")
        return df
    except:
        print(f"Failed loading specified columns... Returning all columns from the {split_name} split")
        df = pd.read_csv(f'{folder}/{split_name}.csv')
        return df
    
train_df = load_data('train', columns=['text', 'stars'],folder='/home/ishanhegde/Desktop/Academics/Spring/COMP 4332/Project 1/data')
valid_df = load_data('valid', columns=['text', 'stars'],folder='/home/ishanhegde/Desktop/Academics/Spring/COMP 4332/Project 1/data')
test_df = load_data('test', columns=['text'],folder='/home/ishanhegde/Desktop/Academics/Spring/COMP 4332/Project 1/data')

train_df['text'][0]


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

model = BertModel.from_pretrained('bert-base-uncased',
                                  output_hidden_states = True, # Whether the model returns all hidden-states.
                                  )

model.eval()

sent_lst=sent_tokenize(train_df['text'][0])

sent_lst

sent_tokenized_ids=[]
sent_segment_ids=[]
for sentence in sent_lst:
    marked_sentence="[CLS] " + sentence + " [SEP]"
    tokenized_sent=tokenizer.tokenize(marked_sentence)
    tokenized_ids=tokenizer.convert_tokens_to_ids(tokenized_sent)
    sent_tokenized_ids.append(tokenized_ids)
    segment_ids = [1]*len(tokenized_sent)
    sent_segment_ids.append(segment_ids)
    
sent_tokenized_ids

sent_embed=[]
sent_embed_sent=[]
review_embed=[]
for i in range(len(sent_tokenized_ids)):
    tokens_tensor=torch.tensor([sent_tokenized_ids[i]])
    segments_tensors = torch.tensor([sent_segment_ids[i]])
    
    with torch.no_grad():
        outputs = model(tokens_tensor, segments_tensors)
        hidden_states = outputs[2]
    
    token_embeddings = torch.stack(hidden_states, dim=0)
    token_embeddings = torch.squeeze(token_embeddings, dim=1)
    token_embeddings = token_embeddings.permute(1,0,2)
    
    token_vecs_sum = []
    for token in token_embeddings:
        sum_vec = torch.sum(token[-4:], dim=0)
        token_vecs_sum.append(sum_vec)
    
    token_vecs = hidden_states[-2][0]
    sent_embed.append(token_vecs_sum)
    sent_embed_sent.append(torch.mean(token_vecs, dim=0))
    review_embed.append(sum(sent_embed_sent)/len(sent_embed_sent))

review_embed
    
len(sent_embed_sent[0])


def make_feature(review):
    sent_lst=sent_tokenize(review)
    sent_tokenized_ids=[]
    sent_segment_ids=[]
    for sentence in sent_lst:
        marked_sentence="[CLS] " + sentence + " [SEP]"
        tokenized_sent=tokenizer.tokenize(marked_sentence)
        tokenized_ids=tokenizer.convert_tokens_to_ids(tokenized_sent)
        sent_tokenized_ids.append(tokenized_ids)
        segment_ids = [1]*len(tokenized_sent)
        sent_segment_ids.append(segment_ids)
    
    #sent_embed=[]
    sent_embed_sent=[]
    review_embed=[]
    for i in range(len(sent_tokenized_ids)):
        tokens_tensor=torch.tensor([sent_tokenized_ids[i]])
        segments_tensors = torch.tensor([sent_segment_ids[i]])
    
        with torch.no_grad():
            outputs = model(tokens_tensor, segments_tensors)
            hidden_states = outputs[2]
    
        #token_embeddings = torch.stack(hidden_states, dim=0)
        #token_embeddings = torch.squeeze(token_embeddings, dim=1)
        #token_embeddings = token_embeddings.permute(1,0,2)
        
        #token_vecs_sum = []
        #for token in token_embeddings:
        #    sum_vec = torch.sum(token[-4:], dim=0)
        #    token_vecs_sum.append(sum_vec)
    
        token_vecs = hidden_states[-2][0]
        #sent_embed.append(token_vecs_sum)
        sent_embed_sent.append(torch.mean(token_vecs, dim=0))
    review_embed.append(sum(sent_embed_sent)/len(sent_embed_sent))
        
    return review_embed[0].numpy()

make_feature(train_df['text'][1])

X_train_1=[]
for i in tqdm.tqdm(range(1416,10000)):
    X_train_1.append(make_feature(train_df['text'][i]))

len(X_train_1)
len(X_train)
temp_x=np.row_stack([X_train,np.array(X_train_1)])
temp_x.shape

train_df.shape

X_train.shape
X_train[0]
y_train=train_df['stars'][:1415]
X_train=np.array(X_train)
X_train



clf = RandomForestClassifier(random_state=42)

clf.fit(X_train,y_train)


clf.predict(make_feature(train_df['text'][1111]).reshape(1,-1))
train_df['text'][1111]
train_df['stars'][1111]

X_test=[]
for i in tqdm.tqdm(range(valid_df.shape[0])):
    X_test.append(make_feature(valid_df['text'][i]))


X_test=np.array(X_test)
y_test=valid_df['stars'][:383]

pred=clf.predict(X_test)

len(pred)

print(classification_report(y_test,pred))

a=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]

len(a[:9])
a[10:]
a[:9]


pd.DataFrame(X_train).to_csv('initial_embeddings.csv',index=False)
