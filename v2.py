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
BLOCK_SIZE = 64
LEARNING_RATE = 3e-4
MAX_NEW_TOKENS = 600
TRAIN_SPLIT = 0.8
VALIDATION_SPLIT = 0.9
MAX_ITER = 20000
EVAL_ITERS = 100
EVAL_INTERVAL = 1000
N_EMBED = 32
NUM_HEADS = 4
NUM_BLOCKS = 4
FFN_MULTIPLIER = 4
DROPOUT = 0.1

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
        self.key = nn.Linear(N_EMBED, head_size, bias=False)
        self.query = nn.Linear(N_EMBED, head_size, bias=False)
        self.value = nn.Linear(N_EMBED, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)  # (B,T,C)
        q = self.query(x)  # (B,T,C)
        wei = q @ k.transpose(-2, -1) * C**-0.5  # (B,T,T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)  # (B,T,T)
        wei= self.dropout(wei)
        v = self.value(x)  # (B,T,C)
        out = wei @ v  # (B,T,C)
        return out

class MultiHeadAttention(nn.Module):
    """Multiple heads of self-attention in parallel."""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads * head_size, N_EMBED)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out

class FeedForward(nn.Module):
    """A simple linear layer followed by a non-linearity."""

    def __init__(self, embed_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, FFN_MULTIPLIER * embed_dim),
            nn.ReLU(),
            nn.Linear(FFN_MULTIPLIER * embed_dim, embed_dim),
            nn.Dropout(DROPOUT),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """Transformer block: communication followed by computation."""

    def __init__(self, embed_dim, num_heads):
        super().__init__()
        head_size = embed_dim // num_heads
        self.sa_heads = MultiHeadAttention(num_heads, head_size)
        self.ffn = FeedForward(embed_dim)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = x + self.sa_heads(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, N_EMBED)
        self.positional_embedding_table = nn.Embedding(BLOCK_SIZE, N_EMBED)
        self.lm_head = nn.Linear(N_EMBED, vocab_size)
        self.sa_heads = MultiHeadAttention(
            num_heads=NUM_HEADS,
            head_size=N_EMBED // NUM_HEADS,
        )
        self.ffn = FeedForward(N_EMBED)
        self.blocks = nn.Sequential(
            *[Block(N_EMBED, num_heads=NUM_HEADS) for _ in range(NUM_BLOCKS)]
        )

    def forward(self, idx, targets=None):
        batch, time = idx.shape
        token_embeddings = self.token_embedding_table(idx)
        position_embeddings = self.positional_embedding_table(
            torch.arange(time, device=idx.device)
        )
        x = token_embeddings + position_embeddings
        x = self.sa_heads(x)
        x = self.ffn(x)
        x = self.blocks(x)
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
            idx_cond = idx[:, -BLOCK_SIZE:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]  # Focus on the last time step.
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


model = BigramLanguageModel(vocab_size=sp.get_piece_size())
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)


for iteration in range(MAX_ITER):
    if iteration % EVAL_INTERVAL == 0:
        losses = estimate_loss()
        print(
            f"step {iteration}: train loss {losses['train']:.4f}, "
            f"val loss {losses['val']:.4f}"
        )
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
