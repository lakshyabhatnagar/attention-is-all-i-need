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
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)
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
            logits, _ = self(idx)
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
