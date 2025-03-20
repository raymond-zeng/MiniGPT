import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm

#hyperparameters
batch_size = 64
block_size = 256
max_iters = 3000
eval_interval = 500
lr = 5e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
embed_size = 384
num_heads = 6
num_layers = 6

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
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


class Head(nn.Module):

    def __init__(self, embed_size, head_size):
        super().__init__()
        self.key = nn.Linear(embed_size, head_size, bias = False)
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
        self.heads = nn.ModuleList([Head(embed_size, head_size) for _ in range(num_heads)])
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

class GPT(nn.Module):

    def __init__(self, vocab_size, embed_size, num_heads, num_layers):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, embed_size)
        self.pos_emb = nn.Embedding(block_size, embed_size)
        self.blocks = nn.Sequential(*[Block(embed_size, num_heads) for _ in range(num_layers)])
        self.norm = nn.RMSNorm(embed_size)
        self.head = nn.Linear(embed_size, vocab_size)

        self.apply(self.__init_weights)
    
    def __init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0, std=0.02)
    
    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_emb(idx)
        pos_emb = self.pos_emb(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.norm(x)
        logits = self.head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss
    
    def generate(self, idx, max_tokens):
        for _ in range(max_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = getBatch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

model = GPT(vocab_size, embed_size, num_heads, num_layers).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')

for i in tqdm.tqdm(range(max_iters)):
    X, Y = getBatch('train')
    logits, loss = model(X, Y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if i % eval_interval == 0 or i == max_iters - 1:
        losses = estimate_loss(model)
        print(f"train loss: {losses['train']:.4f} val loss: {losses['val']:.4f}")

context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(context, max_tokens=500)[0].tolist()))