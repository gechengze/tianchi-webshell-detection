import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import numpy as np
import random
import json
import re

import utils
from dataset import MyDataset
from model import BERT

utils.setup_seed(1234)

train_df = pd.read_csv('train_df.csv')
valid_df = pd.read_csv('valid_df.csv')

vocab = utils.build_vocab(pd.concat([train_df, valid_df], ignore_index=True))

test_df = pd.read_csv('/tcdata/test.csv')
opcode_list = []

for file in tqdm(test_df.file_id):
    with open(f'/tcdata/test/{file}') as f:
        json_str = f.readline()
        opcode_list.append(utils.get_opcode(json_str))

test_df['opcode'] = opcode_list
test_df['len'] = test_df['opcode'].map(lambda x: len(x))

test_df_1 = test_df[test_df['len'] < 128].reset_index(drop=True)
test_df_2 = test_df[(test_df['len'] >= 128) & (test_df['len'] <= 1024)].reset_index(drop=True)
test_df_3 = test_df[test_df['len'] > 1024].reset_index(drop=True)
test_dataset_1 = MyDataset(test_df_1, vocab, max_len=128, truncation=None)
test_dataset_2 = MyDataset(test_df_2, vocab, max_len=512, truncation=None)
test_dataset_3 = MyDataset(test_df_3, vocab, max_len=1024, truncation=None)


def inference(model, test_dataset):
    model = model.to(device)
    model.eval()
    preds = []

    with tqdm(test_dataset, unit='batch') as tepoch:
        for input_ids, attn_mask, _ in tepoch:
            input_ids = input_ids.to(device)
            attn_mask = attn_mask.to(device)
            logits = model(input_ids.unsqueeze(0), attn_mask.unsqueeze(0))
            preds.append(['white', 'black'][logits.argmax(dim=1).item()])

    return preds


model = BERT(num_layers=6, num_heads=6, vocab_size=len(vocab),
             max_len=128, hidden_size=384, ffn_size=384, dropout=0.2)
model.load_state_dict(torch.load('./ckpt/len128_layer6_head6_hidden384_dropout20_maxlen128_valacc9789.pt', map_location=torch.device('cpu')))

test_df_1['prediction'] = inference(model, test_dataset_1)

model = BERT(num_layers=12, num_heads=12, vocab_size=len(vocab),
             max_len=512, hidden_size=768, ffn_size=768, dropout=0.2)
model.load_state_dict(torch.load('./ckpt/len128to1024_layer12_head12_hidden768_dropout20_maxlen512_valacc9577.pt', map_location=torch.device('cpu')))

test_df_2['prediction'] = inference(model, test_dataset_2)

model = BERT(num_layers=12, num_heads=12, vocab_size=len(vocab),
             max_len=1024, hidden_size=768, ffn_size=768, dropout=0.2)
model.load_state_dict(torch.load('./ckpt/len1024_layer12_head12_hidden768_dropout20_maxlen1024_valacc9709.pt', map_location=torch.device('cpu')))

test_df_3['prediction'] = inference(model, test_dataset_3)

df_test_new = pd.concat([test_df_1, test_df_2, test_df_3], ignore_index=True)
df_test_new = df_test_new[['file_id', 'prediction']]
df_test_new = df_test_new.sort_values(['file_id']).reset_index(drop=True)
print(df_test_new.head())
df_test_new.to_csv('./result.csv', index=False)
