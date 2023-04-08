import torch
from tqdm import tqdm
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, df, vocab, max_len, truncation=None):
        super().__init__() 
        df = df.reset_index(drop=True)
        self.sentences = []
        self.attn_mask = []
        self.labels = []
        
        if 'label' not in df.columns: # test data
            df['label'] = 'white'

        for i in tqdm(df.index):
            tokens = [opcode for opcode in df['opcode'][i]]
            if truncation and (len(tokens) < truncation[0] or len(tokens) > truncation[1]):
                continue
            if len(tokens) < max_len - 2:
                pad_len = max_len - 2 - len(tokens)
                self.sentences.append(['<cls>'] + tokens + ['<sep>'] + ['<pad>'] * pad_len)
                self.attn_mask.append([1] * (len(tokens) + 2) + [0] * pad_len)
            else:
                self.sentences.append(['<cls>'] + tokens[:max_len - 2] + ['<sep>'])
                self.attn_mask.append([1] * max_len)
            self.labels.append({'white': 0, 'black': 1}[df['label'][i]])
                
        self.input_ids = [vocab.lookup_indices(tokens) for tokens in self.sentences]
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return torch.LongTensor(self.input_ids[idx]), torch.LongTensor(self.attn_mask[idx]), torch.LongTensor([self.labels[idx]])
