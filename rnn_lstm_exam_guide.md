# RNN/LSTM Exam Revision Guide 🔄
## Sequential Models for Arabic Text Sentiment Analysis

---

## Table of Contents
1. [Sequential Models Fundamentals](#1-sequential-models-fundamentals)
2. [RNN Architecture & Mathematics](#2-rnn-architecture--mathematics)
3. [LSTM Architecture & Mathematics](#3-lstm-architecture--mathematics)
4. [GRU Architecture & Mathematics](#4-gru-architecture--mathematics)
5. [Bidirectional RNNs](#5-bidirectional-rnns)
6. [Text Processing & Embeddings](#6-text-processing--embeddings)
7. [NLP-Specific Preprocessing](#7-nlp-specific-preprocessing)
8. [Sequence Pooling Methods](#8-sequence-pooling-methods)
9. [Advanced Training Techniques](#9-advanced-training-techniques)
10. [Evaluation Metrics for NLP](#10-evaluation-metrics-for-nlp)
11. [Key Differences Summary](#11-key-differences-summary)
12. [PyTorch Implementation Details](#12-pytorch-implementation-details)

---

## 1. Sequential Models Fundamentals

### Why Sequential Models?

Traditional feedforward neural networks (MLPs) have limitations:
- **No memory**: Each input is processed independently
- **Fixed input size**: Cannot handle variable-length sequences
- **No temporal/sequential patterns**: Cannot capture order information

**Sequential models solve this by:**
- Maintaining hidden state across time steps
- Processing variable-length sequences
- Capturing temporal dependencies and patterns

### Applications
- Natural Language Processing (text classification, translation, sentiment analysis)
- Time series prediction (stock prices, weather)
- Speech recognition
- Video analysis
- Music generation

---

## 2. RNN Architecture & Mathematics

### 2.1 Basic RNN Structure

An RNN processes sequences one element at a time, maintaining a **hidden state** that captures information from previous time steps.

**Architecture:**
```
Input: x₁, x₂, x₃, ..., xₜ (sequence of length T)
Hidden: h₀, h₁, h₂, ..., hₜ (hidden states)
Output: y₁, y₂, y₃, ..., yₜ
```

### 2.2 RNN Mathematical Formulas

**At each time step t:**

```
hₜ = tanh(Wₓₕ · xₜ + Wₕₕ · hₜ₋₁ + bₕ)
yₜ = Wₕᵧ · hₜ + bᵧ
```

**Where:**
- `xₜ` = input at time t (dimension: input_size)
- `hₜ` = hidden state at time t (dimension: hidden_size)
- `hₜ₋₁` = previous hidden state
- `h₀` = initial hidden state (usually zeros)
- `Wₓₕ` = input-to-hidden weight matrix (hidden_size × input_size)
- `Wₕₕ` = hidden-to-hidden weight matrix (hidden_size × hidden_size)
- `Wₕᵧ` = hidden-to-output weight matrix (output_size × hidden_size)
- `bₕ, bᵧ` = bias terms
- `tanh` = activation function

**Expanded Formula:**
```
hₜ = tanh(Wₓₕ[xₜ] + Wₕₕ[hₜ₋₁] + bₕ)
   = tanh([x₁ᵗ·w₁ + x₂ᵗ·w₂ + ... + xₙᵗ·wₙ] + [h₁ᵗ⁻¹·u₁ + h₂ᵗ⁻¹·u₂ + ...] + b)
```

### 2.3 RNN Unfolding Through Time

```
t=1:  x₁ → [RNN] → h₁ → y₁
              ↓
t=2:  x₂ → [RNN] → h₂ → y₂
              ↓
t=3:  x₃ → [RNN] → h₃ → y₃
```

**Key Point**: The same weights (Wₓₕ, Wₕₕ, Wₕᵧ) are shared across all time steps.

### 2.4 Backpropagation Through Time (BPTT)

To train RNNs, we use **Backpropagation Through Time**:

1. **Forward pass**: Compute all hidden states h₁, h₂, ..., hₜ
2. **Compute loss**: L = Σ Loss(yₜ, ŷₜ) over all time steps
3. **Backward pass**: Propagate gradients backward through time

**Gradient computation:**
```
∂L/∂Wₕₕ = Σₜ (∂L/∂hₜ · ∂hₜ/∂Wₕₕ)
```

The gradient flows backward through all previous time steps.

### 2.5 Vanishing Gradient Problem ⚠️

**Problem**: When backpropagating through many time steps:

```
∂hₜ/∂h₀ = ∂hₜ/∂hₜ₋₁ · ∂hₜ₋₁/∂hₜ₋₂ · ... · ∂h₁/∂h₀
```

If `|∂hₜ/∂hₜ₋₁| < 1`, the gradient vanishes:
```
(0.5)¹⁰ = 0.00097... → 0
```

**Consequence**: RNN cannot learn long-term dependencies.

**Solution**: LSTM and GRU architectures (see below).

### 2.6 Exploding Gradient Problem

If `|∂hₜ/∂hₜ₋₁| > 1`, gradients explode:
```
(2)¹⁰ = 1024 → ∞
```

**Solution**: Gradient clipping
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

---

## 3. LSTM Architecture & Mathematics

### 3.1 LSTM Overview

**Long Short-Term Memory (LSTM)** solves the vanishing gradient problem by introducing:
- **Memory cell** (Cₜ): Long-term memory
- **Three gates**: Control information flow
  1. **Forget gate** (fₜ): What to forget from memory
  2. **Input gate** (iₜ): What new information to add
  3. **Output gate** (oₜ): What to output

### 3.2 LSTM Mathematical Formulas

**At each time step t:**

```
fₜ = σ(Wₓf·xₜ + Wₕf·hₜ₋₁ + bf)    [Forget gate]
iₜ = σ(Wₓᵢ·xₜ + Wₕᵢ·hₜ₋₁ + bᵢ)    [Input gate]
oₜ = σ(Wₓₒ·xₜ + Wₕₒ·hₜ₋₁ + bₒ)    [Output gate]

C̃ₜ = tanh(Wₓc·xₜ + Wₕc·hₜ₋₁ + bc) [Candidate cell state]

Cₜ = fₜ ⊙ Cₜ₋₁ + iₜ ⊙ C̃ₜ          [New cell state]
hₜ = oₜ ⊙ tanh(Cₜ)                [New hidden state]
```

**Where:**
- `σ` = sigmoid function (outputs 0-1, acts as gate)
- `⊙` = element-wise multiplication (Hadamard product)
- `Cₜ` = cell state (long-term memory)
- `hₜ` = hidden state (short-term output)
- `C̃ₜ` = candidate values to add to memory

### 3.3 LSTM Gates Explained

#### Forget Gate (fₜ)
```
fₜ = σ(Wₓf·xₜ + Wₕf·hₜ₋₁ + bf)
```
- **Range**: [0, 1] due to sigmoid
- **Purpose**: Decides what to forget from Cₜ₋₁
- **fₜ = 0**: Completely forget
- **fₜ = 1**: Completely remember

#### Input Gate (iₜ)
```
iₜ = σ(Wₓᵢ·xₜ + Wₕᵢ·hₜ₋₁ + bᵢ)
C̃ₜ = tanh(Wₓc·xₜ + Wₕc·hₜ₋₁ + bc)
```
- **iₜ**: How much of C̃ₜ to add
- **C̃ₜ**: Candidate values (range: [-1, 1])

#### Cell State Update
```
Cₜ = fₜ ⊙ Cₜ₋₁ + iₜ ⊙ C̃ₜ
```
- First term: What to keep from old memory
- Second term: What new information to add

#### Output Gate (oₜ)
```
oₜ = σ(Wₓₒ·xₜ + Wₕₒ·hₜ₋₁ + bₒ)
hₜ = oₜ ⊙ tanh(Cₜ)
```
- **Purpose**: Decides what to output from Cₜ
- **tanh(Cₜ)**: Squash cell state to [-1, 1]
- **oₜ**: Filter what parts to output

### 3.4 Why LSTM Solves Vanishing Gradients

**Key insight**: The cell state Cₜ has a **linear path** through time:
```
Cₜ = fₜ ⊙ Cₜ₋₁ + iₜ ⊙ C̃ₜ
```

The gradient can flow backward through this linear path without vanishing:
```
∂Cₜ/∂Cₜ₋₁ = fₜ  (element-wise multiplication, not matrix)
```

This allows gradients to flow unchanged if fₜ ≈ 1.

### 3.5 LSTM Visualization

```
          ┌─────────────────────────────┐
          │                             │
xₜ, hₜ₋₁ →│  fₜ    iₜ    C̃ₜ    oₜ     │→ hₜ
          │  ↓     ↓     ↓     ↓      │
   Cₜ₋₁ →│  ×  +  ×  =  Cₜ  → tanh → ×│→ hₜ
          │    ↖_____↗         ↓      │
          │                    └──────┘│
          └─────────────────────────────┘
```

---

## 4. GRU Architecture & Mathematics

### 4.1 GRU Overview

**Gated Recurrent Unit (GRU)** is a simplified version of LSTM:
- **Fewer parameters**: Faster training, less overfitting
- **Two gates instead of three**:
  1. **Reset gate** (rₜ): How much past information to forget
  2. **Update gate** (zₜ): How much to update hidden state
- **No separate cell state**: Only hidden state hₜ

### 4.2 GRU Mathematical Formulas

```
zₜ = σ(Wₓz·xₜ + Wₕz·hₜ₋₁ + bz)    [Update gate]
rₜ = σ(Wₓᵣ·xₜ + Wₕᵣ·hₜ₋₁ + bᵣ)    [Reset gate]

h̃ₜ = tanh(Wₓₕ·xₜ + Wₕₕ·(rₜ ⊙ hₜ₋₁) + bₕ)  [Candidate hidden state]

hₜ = (1 - zₜ) ⊙ hₜ₋₁ + zₜ ⊙ h̃ₜ     [New hidden state]
```

### 4.3 GRU Gates Explained

#### Update Gate (zₜ)
```
zₜ = σ(Wₓz·xₜ + Wₕz·hₜ₋₁ + bz)
```
- **Purpose**: Balance between old and new information
- **zₜ = 0**: Keep old state (hₜ = hₜ₋₁)
- **zₜ = 1**: Use new state (hₜ = h̃ₜ)

#### Reset Gate (rₜ)
```
rₜ = σ(Wₓᵣ·xₜ + Wₕᵣ·hₜ₋₁ + bᵣ)
h̃ₜ = tanh(Wₓₕ·xₜ + Wₕₕ·(rₜ ⊙ hₜ₋₁) + bₕ)
```
- **Purpose**: Decides how much past information to use when computing h̃ₜ
- **rₜ = 0**: Ignore past (h̃ₜ computed only from xₜ)
- **rₜ = 1**: Use full past information

#### Hidden State Update
```
hₜ = (1 - zₜ) ⊙ hₜ₋₁ + zₜ ⊙ h̃ₜ
```
- **Interpolation** between old and new state
- If zₜ = 0.3: hₜ = 0.7·hₜ₋₁ + 0.3·h̃ₜ

### 4.4 GRU vs LSTM

| Aspect | LSTM | GRU |
|--------|------|-----|
| **Gates** | 3 (forget, input, output) | 2 (reset, update) |
| **States** | Cell (Cₜ) + Hidden (hₜ) | Only Hidden (hₜ) |
| **Parameters** | More (~4×) | Fewer (~3×) |
| **Speed** | Slower | Faster |
| **Performance** | Slightly better on complex tasks | Similar on most tasks |
| **Overfitting** | More prone (more params) | Less prone |

**Rule of thumb:**
- Use LSTM for: Complex sequences, large datasets
- Use GRU for: Faster training, smaller datasets

---

## 5. Bidirectional RNNs

### 5.1 Motivation

**Problem**: Standard RNNs only see past context.

**Example**: "The animal didn't cross the street because it was too ___"
- Forward RNN: Only sees words before "___"
- To predict correctly, we need future context ("tired" vs "wide")

### 5.2 Bidirectional Architecture

```
Forward:  x₁ → h₁ᶠ → h₂ᶠ → h₃ᶠ → h₄ᶠ
                               ↓
Backward: x₁ ← h₁ᵇ ← h₂ᵇ ← h₃ᵇ ← h₄ᵇ
          ↓    ↓     ↓     ↓     ↓
Output:   y₁   y₂    y₃    y₄    y₅
```

### 5.3 Mathematical Formulas

**Forward pass:**
```
h₁ᶠ, h₂ᶠ, ..., hₜᶠ = RNN_forward(x₁, x₂, ..., xₜ)
```

**Backward pass:**
```
h₁ᵇ, h₂ᵇ, ..., hₜᵇ = RNN_backward(xₜ, xₜ₋₁, ..., x₁)
```

**Concatenate:**
```
hₜ = [hₜᶠ ; hₜᵇ]  (dimension: 2 × hidden_size)
```

**Output:**
```
yₜ = Wᵧ · hₜ + bᵧ
```

### 5.4 Benefits & Trade-offs

**Benefits:**
- Captures both past and future context
- **Better performance** on most NLP tasks
- Essential for tasks like Named Entity Recognition, POS tagging

**Trade-offs:**
- **2× parameters**: double the hidden state size
- **Cannot do online prediction**: needs entire sequence
- **Slower training**: processes sequence twice

**From your code:**
```python
self.rnn = nn.LSTM(
    embed_dim, hidden_size,
    bidirectional=True  # ← Enables Bi-LSTM
)
out_dim = hidden_size * 2  # ← Double size for concatenation
```

---

## 6. Text Processing & Embeddings

### 6.1 Word Embeddings

**Problem**: Neural networks need numerical inputs, but text is discrete.

**Solution**: Map words to dense vectors (embeddings).

**Example:**
```
"hello" → [0.2, 0.5, -0.1, 0.8]
"world" → [0.3, 0.4, -0.2, 0.7]
```

### 6.2 Embedding Layer Mathematics

```
E = Embedding Matrix (vocab_size × embed_dim)
x = word index (integer)

embedding(x) = E[x, :]  (row lookup)
```

**Example:**
```
Vocabulary: ["<PAD>", "<UNK>", "hello", "world", "!"]
vocab_size = 5
embed_dim = 3

E = [[0.0, 0.0, 0.0],  # <PAD>
     [0.1, 0.1, 0.1],  # <UNK>
     [0.5, 0.2, 0.8],  # "hello"
     [0.6, 0.3, 0.7],  # "world"
     [0.2, 0.9, 0.4]]  # "!"

Input: [2, 3, 4] = ["hello", "world", "!"]
Output: [[0.5, 0.2, 0.8],
         [0.6, 0.3, 0.7],
         [0.2, 0.9, 0.4]]
```

**Key properties:**
- **Learnable**: E is updated during training
- **Shared**: Same embedding used throughout the model
- **Dense**: Low-dimensional representation (typically 50-300 dims)

### 6.3 Padding

**Problem**: Sequences have different lengths.

**Solution**: Pad short sequences to a fixed length.

```
Original: ["hello", "world"]
Padded:   ["hello", "world", "<PAD>", "<PAD>", "<PAD>"]

Indices:  [2, 3, 0, 0, 0]
```

**In PyTorch:**
```python
self.embedding = nn.Embedding(
    vocab_size, 
    embed_dim, 
    padding_idx=0  # ← Don't update <PAD> embedding
)
```

### 6.4 Tokenization

**Process**: Convert text to sequence of indices.

```
Text: "مرحبا بالعالم"
     ↓ Tokenize
Tokens: ["مرحبا", "بالعالم"]
     ↓ Encode
Indices: [234, 567]
```

**From your code:**
```python
class ArabicTokenizer:
    def fit(self, texts):
        # Build vocabulary from training data
        words = []
        for text in texts:
            words.extend(text.split())
        
        # Keep top vocab_size most common words
        counts = Counter(words)
        for i, (word, _) in enumerate(counts.most_common(vocab_size - 2)):
            self.word2idx[word] = i + 2  # Reserve 0, 1 for special tokens
    
    def encode(self, text, max_len=64):
        # Convert text to indices
        ids = [self.word2idx.get(w, 1) for w in text.split()[:max_len]]
        # Pad to max_len
        return ids + [0] * (max_len - len(ids))
```

---

## 7. NLP-Specific Preprocessing

### 7.1 Arabic Text Preprocessing

**Challenges in Arabic:**
1. Different forms of same letter (أ، إ، آ → ا)
2. Diacritics (تشكيل): َ ُ ِ ّ ْ ٰ
3. Tatweel (elongation): ـــ
4. Multiple forms of letters at word end (ة vs ه, ى vs ي)

**Normalization steps (from your code):**
```python
def arabic_preprocess(text):
    # 1. Remove URLs
    text = re.sub(r'http\S+|www\.\S+', '', text)
    
    # 2. Remove numbers (Arabic & English)
    text = re.sub(r'[0-9٠-٩]+', '', text)
    
    # 3. Keep only Arabic letters and spaces
    text = re.sub(r'[^\w\s\u0600-\u06FF]', ' ', text)
    
    # 4. Normalize Alef forms
    text = re.sub(r'[إأآا]', 'ا', text)
    
    # 5. Normalize Ya
    text = re.sub(r'ى', 'ي', text)
    
    # 6. Normalize Ta Marbuta
    text = re.sub(r'ة', 'ه', text)
    
    # 7. Remove diacritics
    text = re.sub(r'[ًٌٍَُِّْٰ]', '', text)
    
    # 8. Remove tatweel
    text = re.sub(r'ـ+', '', text)
    
    # 9. Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text
```

### 7.2 Why Preprocessing Matters

**Example:**
```
Before: "السَّـــــلامُ عَليْكُم"
After:  "السلام عليكم"

Before: "مُحَمَّد"
After:  "محمد"
```

**Benefits:**
- Reduces vocabulary size
- Improves generalization
- Handles spelling variations
- Removes noise (URLs, numbers)

---

## 8. Sequence Pooling Methods

After processing the sequence through RNN/LSTM, we get outputs at all time steps. For classification, we need a **single vector** representation.

### 8.1 Last Output (Default)
```
h₁, h₂, h₃, ..., hₜ → Use hₜ only
```
- **Pros**: Simple, captures final state
- **Cons**: Ignores earlier information, sensitive to padding

### 8.2 Mean Pooling ⭐ (Your code uses this)
```
h_pooled = (h₁ + h₂ + h₃ + ... + hₜ) / T
```

**Formula:**
```
h_pooled = Σᵢ hᵢ / T
```

**With masking (ignore padding):**
```python
mask = (x != 0).unsqueeze(-1).float()  # 1 for real tokens, 0 for padding
h_pooled = (out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
```

**Example:**
```
Sequence: ["hello", "world", "<PAD>", "<PAD>"]
Mask:     [1, 1, 0, 0]
Outputs:  [h₁, h₂, h₃, h₄]

h_pooled = (h₁ + h₂) / 2  (ignores h₃, h₄)
```

**Benefits:**
- Considers all tokens equally
- Robust to padding
- Better for sentiment analysis (all words contribute)

### 8.3 Max Pooling
```
h_pooled = max(h₁, h₂, ..., hₜ) element-wise
```
- **Pros**: Captures strongest features
- **Cons**: Can ignore important information

### 8.4 Attention Pooling
```
α₁, α₂, ..., αₜ = Attention(h₁, h₂, ..., hₜ)
h_pooled = α₁·h₁ + α₂·h₂ + ... + αₜ·hₜ
```
- **Pros**: Learns which tokens are important
- **Cons**: More complex, more parameters

---

## 9. Advanced Training Techniques

### 9.1 Gradient Clipping

**Problem**: Exploding gradients in RNNs.

**Solution**: Clip gradient norm to maximum value.

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

**Mathematical formula:**
```
If ||g|| > max_norm:
    g_clipped = g × (max_norm / ||g||)
else:
    g_clipped = g
```

Where `||g||` is the L2 norm of all gradients.

**Why it works:**
- Limits gradient magnitude
- Prevents explosive updates
- Allows training to converge

### 9.2 Learning Rate Scheduling

**ReduceLROnPlateau** (from your code):
```python
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    patience=3,    # Wait 3 epochs
    factor=0.5     # Multiply LR by 0.5
)
```

**How it works:**
```
Epoch 1: LR = 0.001, Val Loss = 0.5
Epoch 2: LR = 0.001, Val Loss = 0.48
Epoch 3: LR = 0.001, Val Loss = 0.47
Epoch 4: LR = 0.001, Val Loss = 0.46  ← Still improving
Epoch 5: LR = 0.001, Val Loss = 0.46  ← No improvement (1/3)
Epoch 6: LR = 0.001, Val Loss = 0.46  ← No improvement (2/3)
Epoch 7: LR = 0.001, Val Loss = 0.46  ← No improvement (3/3)
Epoch 8: LR = 0.0005 ← Reduced! (0.001 × 0.5)
```

**Benefits:**
- Automatic adaptation
- Fine-tunes in later stages
- Helps escape plateaus

### 9.3 Early Stopping

**Algorithm:**
```python
class EarlyStopping:
    def __init__(self, patience=7):
        self.patience = patience
        self.counter = 0
        self.best_loss = None
        self.best_model = None
    
    def __call__(self, val_loss, model):
        if self.best_loss is None or val_loss < self.best_loss - 0.001:
            # Improvement found
            self.best_loss = val_loss
            self.best_model = copy of model
            self.counter = 0
        else:
            # No improvement
            self.counter += 1
            if self.counter >= self.patience:
                return True  # Stop training
        return False
```

**Example:**
```
Epoch 1: Val Loss = 0.500 → Save (best so far)
Epoch 2: Val Loss = 0.450 → Save (improved!)
Epoch 3: Val Loss = 0.445 → Save
Epoch 4: Val Loss = 0.446 → No save (counter = 1)
Epoch 5: Val Loss = 0.447 → No save (counter = 2)
...
Epoch 10: Val Loss = 0.450 → No save (counter = 7) → STOP!
```

### 9.4 Layer Normalization

**Formula:**
```
y = γ × (x - μ) / √(σ² + ε) + β
```

Where:
- μ = mean of x
- σ² = variance of x
- γ, β = learnable parameters
- ε = small constant (1e-5) for numerical stability

**Applied to embeddings (from your code):**
```python
emb = self.embedding(x)          # (batch, seq_len, embed_dim)
emb = self.layer_norm(emb)       # Normalize across embed_dim
```

**Benefits:**
- Stabilizes training
- Allows higher learning rates
- Reduces internal covariate shift

### 9.5 Weight Decay (L2 Regularization)

**AdamW optimizer** (from your code):
```python
optimizer = torch.optim.AdamW(
    model.parameters(), 
    lr=1e-3, 
    weight_decay=1e-4  # ← L2 regularization
)
```

**Effect:**
```
Loss = MSE_loss + λ × Σ(w²)
      = MSE_loss + 1e-4 × Σ(w²)
```

**Gradient update:**
```
w = w - lr × (∂MSE/∂w + 2λw)
  = w - 1e-3 × (∂MSE/∂w + 2×1e-4×w)
```

---

## 10. Evaluation Metrics for NLP

### 10.1 Regression Metrics (Sentiment Scores)

Your task predicts continuous scores (e.g., 1-10 rating).

#### Mean Absolute Error (MAE)
```
MAE = (1/n) Σ |y_true - y_pred|
```
- **Interpretation**: Average absolute difference
- **Lower is better**
- **Example**: MAE = 0.5 means predictions are off by 0.5 points on average

#### Root Mean Squared Error (RMSE)
```
RMSE = √[(1/n) Σ (y_true - y_pred)²]
```
- Penalizes large errors more than MAE
- Same units as target variable

#### R² Score
```
R² = 1 - (SS_residual / SS_total)
   = 1 -