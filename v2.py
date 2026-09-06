import sentencepiece as spm
import torch
import torch.nn as nn
from torch.nn import functional as F

INPUT_FILE = "input.txt"
MODEL_PREFIX = "m"
MODEL_FILE = "m.model"

# Hyperparameters.
VOCAB_SIZE = 1000
SEED = 1337
BATCH_SIZE = 32
BLOCK_SIZE = 8
LEARNING_RATE = 1e-3
MAX_NEW_TOKENS = 600
TRAIN_SPLIT = 0.8
VALIDATION_SPLIT = 0.9
MAX_ITER=10000
EVAL_ITERS=200
n_embed = 32  # Size of the embedding vector for each token.

# Train a SentencePiece tokenizer.
spm.SentencePieceTrainer.train(
    input=INPUT_FILE,
    model_prefix=MODEL_PREFIX,
    vocab_size=VOCAB_SIZE,
)

sp = spm.SentencePieceProcessor(model_file=MODEL_FILE)

with open(INPUT_FILE, encoding="utf-8") as file:
    text = file.read()
# Transform the input data into tokens.
token_ids = sp.encode(text, out_type=int)
data_tokens = torch.tensor(token_ids, dtype=torch.long)

n = len(data_tokens)
train_end = int(TRAIN_SPLIT * n)
val_end = int(VALIDATION_SPLIT * n)

train_data = data_tokens[:train_end]
val_data = data_tokens[train_end:val_end]
test_data = data_tokens[val_end:]


def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([data[i : i + BLOCK_SIZE] for i in ix])
    y = torch.stack([data[i + 1 : i + BLOCK_SIZE + 1] for i in ix])
    return x, y


torch.manual_seed(SEED)

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    """One head of self-attention."""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)  # (B,T,C)
        q = self.query(x)  # (B,T,C)
        wei = q @ k.transpose(-2, -1) * C**-0.5  # (B,T,T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)  # (B,T,T)
        v = self.value(x)  # (B,T,C)
        out = wei @ v  # (B,T,C)
        return out

class MultiHeadAttention(nn.Module):
    """Multiple heads of self-attention in parallel."""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads * head_size, n_embed)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out

class FeedForward(nn.Module):
    """A simple linear layer followed by a non-linearity."""

    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """Transformer block: communication followed by computation."""

    def __init__(self, n_embed, num_heads):
        super().__init__()
        head_size = n_embed // num_heads
        self.sa_heads = MultiHeadAttention(num_heads, head_size)
        self.ffn = FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.sa_heads(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x
    
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.positional_embedding_table = nn.Embedding(BLOCK_SIZE, n_embed)
        self.lm_head=nn.Linear(n_embed, vocab_size)
        self.sa_heads=MultiHeadAttention(num_heads=4, head_size=n_embed//4) #four heads of self attention
        self.ffn=FeedForward(n_embed) #feed forward network
        self.blocks=nn.Sequential(*[Block(n_embed, num_heads=4) for _ in range(4)]) #stack of 4 transformer blocks

    def forward(self, idx, targets=None):
        B,T=idx.shape
        token_embeddings=self.token_embedding_table(idx) #(B,T,C)
        position_embeddings=self.positional_embedding_table(torch.arange(T, device=idx.device)) #(T,C)
        x=token_embeddings+position_embeddings #(B,T,vocab_size)
        x=self.sa_heads(x) #apply self attention
        x=self.ffn(x) #apply feed forward network
        x=self.blocks(x) #apply transformer blocks
        logits = self.lm_head(x)
        if targets is None:
            loss = None
        else:
            batch, time, channels = logits.shape
            logits = logits.view(batch * time, channels)
            targets = targets.view(batch * time)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is a (B, T) array of token indices in the current context.
        for _ in range(max_new_tokens):
            idx_cond=idx[:, -BLOCK_SIZE:] # crop idx to the last BLOCK_SIZE tokens
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]  # Focus on the last time step.
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


model = BigramLanguageModel(vocab_size=sp.get_piece_size())
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)



for iter in range(MAX_ITER):
    if iter % 1000 == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    xb, yb = get_batch("train")
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(loss.item())

print(estimate_loss())

idx = model.generate(
    idx=torch.zeros((1, 1), dtype=torch.long),
    max_new_tokens=MAX_NEW_TOKENS,
)
print(sp.decode(idx[0].tolist()))
