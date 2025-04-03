# Unigram and Bigram Language Models

A python implementation of unigram and bigram language models for language processing. This repository includes modular implementations using Laplace smoothing with optional automatic optimization.

## Usage

### Unigram Model

```python
import unigram

# Create and train a unigram model
train_data = ["how", "are", "you", "today"]
model = unigram.UnigramLM()
model.train(train_data, alpha=1)

# Calculate log probability of a sequence
test_data = ["how", "are", "you", "doing"]
log_prob = model.log_probability(test_data)
print(f"Log probability: {log_prob}")
```

### Bigram Model

```python
import bigram

# Create and train a bigram model
train_data = [
    ["*START*", "how", "are", "you", "*STOP*"],
    ["*START*", "what's", "your", "name", "*STOP*"]
]
model = bigram.BigramLM()
model.train(train_data, alpha=1, beta=1)

# Calculate log probability of a sequence
test_data = [["*START*", "how", "are", "you", "today", "*STOP*"]]
log_prob = model.log_probability(test_data)
print(f"Log probability: {log_prob}")
```
