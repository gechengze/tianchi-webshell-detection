import math
import torch
from torch import nn
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 


class Embedding(nn.Module):
    def __init__(self, vocab_size, max_len, hidden_size):
        super(Embedding, self).__init__()
        self.tok_embed = nn.Embedding(vocab_size, hidden_size)
        self.pos_embed = nn.Embedding(max_len, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x):
        seq_len = x.shape[1]
        pos = torch.arange(seq_len, dtype=torch.long).unsqueeze(0).expand_as(x)
        pos = pos.to(device)
        embedded = self.tok_embed(x) + self.pos_embed(pos)
        return self.norm(embedded)
    

class ScaledDotProductionAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductionAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, query, key, value, attn_mask):
        d_k = key.shape[-1]
        scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(d_k)
        scores.masked_fill_(attn_mask, -1e9)
        attn_weights = self.softmax(scores)
        return torch.matmul(attn_weights, value) 

    
class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.hidden_size = hidden_size 
        self.num_heads = num_heads 
        self.key_size = self.value_size = self.hidden_size // self.num_heads 
        self.attention = ScaledDotProductionAttention()
        self.W_Q = nn.Linear(hidden_size, hidden_size) 
        self.W_K = nn.Linear(hidden_size, hidden_size)  
        self.W_V = nn.Linear(hidden_size, hidden_size) 
        self.fc = nn.Linear(hidden_size, hidden_size)  
        
    def forward(self, query, key, value, attn_mask):
        batch_size = query.shape[0]
        seq_len = query.shape[1]

        q_s = self.W_Q(query).reshape(batch_size, -1, self.num_heads, self.key_size).transpose(1, 2)
        k_s = self.W_Q(key).reshape(batch_size, -1, self.num_heads, self.key_size).transpose(1, 2)
        v_s = self.W_Q(value).reshape(batch_size, -1, self.num_heads, self.value_size).transpose(1, 2)

        attn_mask = attn_mask.data.eq(0).unsqueeze(1).expand(batch_size, seq_len, seq_len)
        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
        context = self.attention(q_s, k_s, v_s, attn_mask)

        context = context.transpose(1, 2).reshape(batch_size, -1, self.hidden_size)

        output = self.fc(context)

        return output
    
    
def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class PositionWiseFFN(nn.Module):
    def __init__(self, hidden_size, ffn_size):
        super(PositionWiseFFN, self).__init__()
        self.fc1 = nn.Linear(hidden_size, ffn_size)
        self.fc2 = nn.Linear(ffn_size, hidden_size)

    def forward(self, x):
        return self.fc2(gelu(self.fc1(x)))
    

class AddNorm(nn.Module):
    def __init__(self, norm_shape, dropout):
        super(AddNorm, self).__init__()
        self.layer_norm = nn.LayerNorm(norm_shape)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, y):
        return self.layer_norm(x + self.dropout(y))
    
    
class EncoderBlock(nn.Module):
    def __init__(self, num_heads, hidden_size, ffn_size, dropout):
        super(EncoderBlock, self).__init__()
        self.attention = MultiHeadAttention(hidden_size=hidden_size, num_heads=num_heads)
        self.add_norm1 = AddNorm(norm_shape=hidden_size, dropout=dropout)
        self.ffn = PositionWiseFFN(hidden_size=hidden_size, ffn_size=ffn_size)
        self.add_norm2 = AddNorm(norm_shape=hidden_size, dropout=dropout)

    def forward(self, x, attn_mask):
        output = self.add_norm1(x, self.attention(x, x, x, attn_mask))
        output = self.add_norm2(output, self.ffn(output))
        return output
    

class BERT(nn.Module):
    def __init__(self, num_layers, num_heads, vocab_size, max_len, hidden_size, ffn_size, dropout):
        super(BERT, self).__init__()

        self.embedding = Embedding(vocab_size=vocab_size, max_len=max_len, hidden_size=hidden_size)

        self.layers = nn.Sequential()

        for i in range(num_layers):
            self.layers.add_module(f'{i}', EncoderBlock(num_heads=num_heads, hidden_size=hidden_size,
                                                        ffn_size=ffn_size, dropout=dropout))
            
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, input_ids, attn_mask):
        output = self.embedding(input_ids)

        for layer in self.layers:
            output = layer(output, attn_mask)

        cls_output = output[:, 0]
        
        return self.fc(cls_output)