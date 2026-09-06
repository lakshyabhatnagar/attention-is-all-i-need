# attention-is-all-i-need

This is a small, deliberately hands-on language-model project.

I started it to answer one question:

> How much do I actually understand attention?

I think the answer is: pretty much. I can now follow the data from text to
tokens, through queries, keys, values, masking, residual connections, and
back to generated text. The code is intentionally small and educational, not
a production-ready transformer.

## The progression

- `bigram.py` is the starting point. It uses SentencePiece tokenization and a
  token-to-token embedding table to predict the next token.
- `v2.py` is the attention version. It keeps the implementation close to the
  underlying ideas instead of hiding them behind a transformer library.

The `BigramLanguageModel` name remains in `v2.py` because it came from the
first version. The second version is really a small decoder-style transformer.

## How `v2.py` works

1. SentencePiece builds a 1,000-token vocabulary from `input.txt`.
2. The corpus is converted into integer token IDs and split into 80% training,
   10% validation, and 10% test data.
3. `get_batch()` samples random 64-token windows. The input is the window and
   the target is the same window shifted by one token.
4. The model adds token embeddings and positional embeddings.
5. Multi-head causal self-attention is applied. Each head creates keys,
   queries, and values, scales the query-key scores, masks future positions,
   applies softmax, and combines the values.
6. Feed-forward layers and four residual transformer blocks process the
   sequence.
7. Cross-entropy loss is optimized with AdamW.
8. Generation repeatedly predicts one token, appends it, and keeps only the
   latest context window.

## Techniques used

- SentencePiece subword tokenization
- Token and positional embeddings
- Causal lower-triangular attention masking
- Four-head self-attention
- Query/key scaling and softmax attention weights
- Pre-layer-normalized residual blocks
- Feed-forward networks with ReLU
- Dropout for regularization
- AdamW optimization
- Periodic train/validation loss estimation with `torch.no_grad()`

## Current `v2.py` settings

| Setting | Value |
| --- | ---: |
| Vocabulary size | 1,000 |
| Embedding size | 32 |
| Attention heads | 4 |
| Transformer blocks | 4 |
| Context length | 64 tokens |
| Feed-forward expansion | 4x |
| Batch size | 32 |
| Dropout | 0.1 |
| Learning rate | 0.0003 |
| Training iterations | 20,000 |
| Evaluation batches | 100 every 1,000 iterations |
| Random seed | 1337 |
| Generated tokens | 600 |

The baseline in `bigram.py` uses an 8-token context, a learning rate of
`0.001`, and 10,000 training steps.

## Running it

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch sentencepiece
python v2.py
```

The script expects `input.txt` to be present. It trains the tokenizer when it
runs and writes `m.model` and `m.vocab` using the configured model prefix.

## What this project is — and is not

This is me building the pieces myself to make sure I understand them. It is
not trying to compete with mature transformer implementations. There is no
checkpointing, scheduler, device abstraction, or polished experiment setup,
and the test split is created but not evaluated yet. Those are future project
ideas, not prerequisites for answering the original question.
