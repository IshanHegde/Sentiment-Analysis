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

