import torch
import torch.nn as nn
import torch.nn.functional as F

#hyperparameters
batch_size = 64
block_size = 256
max_iters = 3000
eval_interval = 500
lr = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200

torch.manual_seed(0)

with open('input.txt', 'r') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}
encode = lambda s : [char_to_idx[ch] for ch in s]
decode = lambda x : ''.join([idx_to_char[i] for i in x])

data = torch.tensor(encode(text), dtype=torch.long, device=device)
n = int(0.9 * data.size(0))
train_data, val_data = data[:n], data[n:]

def getBatch(split):
    if split == 'train':
        data = train_data
    else:
        data = val_data
    start_points = torch.randint(0, data.size(0) - block_size, (batch_size,))
    X = [data[start:start + block_size] for start in start_points]
    Y = [data[start + 1:start + block_size + 1] for start in start_points]
    return torch.stack(X).to(device), torch.stack(Y).to(device)


class Head(nn.Module):
    def __init__(self, embed_size, head_size):
        super().__init__()
        self.key = nn.Embedding(embed_size, head_size, bias = False)
        self.query = nn.Linear(embed_size, head_size, bias = False)
        self.value = nn.Linear(embed_size, head_size, bias = False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        B, T, C = x.size()
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        weights = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        weights = F.softmax(weights, dim=-1)
        weights = self.dropout(weights)
        return weights @ v

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(embed_size, head_size) for _ in range(head_size)])
        self.proj = nn.Linear(head_size * num_heads, embed_size)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        out =  self.proj(torch.cat([head(x) for head in self.heads], dim=-1))
        return self.dropout(out)

class Block(nn.Module):
    def __init__(self, embed_size, num_heads):
        super().__init__()
        head_size = embed_size // num_heads
        self.attention = MultiHeadAttention(embed_size, num_heads, head_size)
        self.net = nn.Sequential(
            nn.Linear(embed_size, 4 * embed_size),
            nn.GELU(),
            nn.Linear(4 * embed_size, embed_size),
            nn.Dropout(0.2)
        )
        self.norm1 = nn.RMSNorm(embed_size)
        self.norm2 = nn.RMSNorm(embed_size)
    
    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.net(self.norm2(x))
        return x