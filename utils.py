import torch
import json
import re
import numpy as np
import random
import torchtext
from tqdm import tqdm
from collections import Counter


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    
def get_opcode(json_str):
    json_obj = json.loads(json_str)    
    opcode = []

    def dfs(json_obj):
        if json_obj['name'] == 'NULL':
            return
        opcode.append(re.findall('\[(.*?):', json_obj['name'])[0])

        if len(json_obj['children']) == 0:
            return
        for json_obj in json_obj['children']:
            dfs(json_obj)

    dfs(json_obj)
    return opcode
    

def build_vocab(df):
    print('Start building vocab...')
    counter = Counter()
    for source_code in tqdm(df['opcode']):
        tokens = [code for code in eval(source_code)]
        counter.update(tokens)

    vocab = torchtext.vocab.vocab(counter, specials=['<unk>', '<pad>', '<cls>', '<sep>'])
    vocab.set_default_index(vocab['<unk>'])
    print('Finish building vocab!')
    return vocab