# 🧠 Synthesis: Deep Learning for Natural Language Processing

## Lab 3 - Sequence Models & Transformers

**Course:** Deep Learning  
**Institution:** Université Abdelmalek Essaadi - Faculty of Sciences and Techniques of Tangier  
**Department:** Computer Engineering  
**Program:** Master MBD  
**Professor:** Pr. ELAACHAK LOTFI

---

## 📋 Table of Contents

1. [Introduction](#1-introduction)
2. [Part 1: Text Classification with Sequence Models](#2-part-1-text-classification-with-sequence-models)
   - [2.1 Data Collection](#21-data-collection)
   - [2.2 NLP Preprocessing Pipeline](#22-nlp-preprocessing-pipeline)
   - [2.3 Sequence Models Architecture](#23-sequence-models-architecture)
   - [2.4 Training & Hyperparameter Tuning](#24-training--hyperparameter-tuning)
   - [2.5 Evaluation Metrics](#25-evaluation-metrics)
3. [Part 2: Transformers for Text Generation](#3-part-2-transformers-for-text-generation)
   - [3.1 GPT-2 Architecture](#31-gpt-2-architecture)
   - [3.2 Fine-tuning Process](#32-fine-tuning-process)
   - [3.3 Text Generation Techniques](#33-text-generation-techniques)
4. [Mathematical Foundations](#4-mathematical-foundations)
5. [Key Concepts Reminder](#5-key-concepts-reminder)
6. [Practical Implementation Guide](#6-practical-implementation-guide)
7. [Conclusion](#7-conclusion)
8. [References](#8-references)

---

## 1. Introduction

This synthesis documents the comprehensive learning outcomes from Lab 3, focusing on building deep neural network architectures for Natural Language Processing (NLP) using PyTorch and Sequence Models. The lab covered two main objectives:

1. **Classification Task**: Building and comparing RNN, BiRNN, GRU, and LSTM models for Arabic text relevance scoring
2. **Text Generation**: Fine-tuning GPT-2 transformer for generating contextual paragraphs

### Key Learning Objectives

- Understanding recurrent neural network architectures and their mathematical foundations
- Implementing NLP preprocessing pipelines for Arabic text
- Training and evaluating sequence models with proper hyperparameter tuning
- Leveraging pre-trained transformer models for text generation tasks

---

## 2. Part 1: Text Classification with Sequence Models

### 2.1 Data Collection

#### Web Scraping Methodology

Data collection was performed using Python web scraping libraries to gather Arabic text data from multiple sources on a specific topic (Technology & AI).

**Tools Comparison:**

| Library | Use Case | Advantages | Limitations |
|---------|----------|------------|-------------|
| BeautifulSoup | Simple HTML parsing | Easy syntax, good documentation | No built-in crawling |
| Scrapy | Large-scale scraping | Fast, asynchronous, robust | Steeper learning curve |

**Dataset Structure:**

| Column | Type | Description |
|--------|------|-------------|
| Text | String | Arabic language content |
| Score | Float | Relevance score (0-10) |

#### 💡 Key Concepts Reminder: Web Scraping

> **HTTP Request/Response Cycle**: Web scraping involves sending HTTP GET requests to servers and parsing the HTML response. Understanding DOM structure is essential for extracting specific elements using CSS selectors or XPath expressions.

---

### 2.2 NLP Preprocessing Pipeline

Arabic text preprocessing requires specialized handling due to unique linguistic characteristics.

#### Pipeline Architecture

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌────────────┐
│  Raw Text   │───▶│   Cleaning   │───▶│ Normalization│───▶│ Tokenization│
└─────────────┘    └──────────────┘    └─────────────┘    └────────────┘
                                                                 │
                                                                 ▼
┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌────────────┐
│   Output    │◀───│   Stemming   │◀───│ Stop Words  │◀───│   Tokens   │
└─────────────┘    └──────────────┘    └─────────────┘    └────────────┘
```

#### Preprocessing Steps

| Step | Operation | Example |
|------|-----------|---------|
| URL Removal | `re.sub(r'http\S+', '', text)` | `http://...` → ∅ |
| Number Removal | `re.sub(r'[0-9٠-٩]+', '', text)` | `123`, `١٢٣` → ∅ |
| Alef Normalization | `re.sub(r'[إأآا]', 'ا', text)` | `إ أ آ` → `ا` |
| Yaa Normalization | `re.sub(r'ى', 'ي', text)` | `ى` → `ي` |
| Diacritics Removal | `re.sub(r'[ًٌٍَُِّْ]', '', text)` | `مُحَمَّد` → `محمد` |
| Tatweel Removal | `re.sub(r'ـ+', '', text)` | `مـــرحبا` → `مرحبا` |

#### 💡 Key Concepts Reminder: Text Normalization

> **Character Encoding**: Arabic text uses Unicode (UTF-8) with characters in the range U+0600 to U+06FF. Normalization ensures consistent representation by mapping variant forms to a canonical form, reducing vocabulary size and improving model generalization.

---

### 2.3 Sequence Models Architecture

#### 2.3.1 Recurrent Neural Network (RNN)

**Architecture Overview:**

```
     x₁         x₂         x₃         x₄
      │          │          │          │
      ▼          ▼          ▼          ▼
   ┌─────┐    ┌─────┐    ┌─────┐    ┌─────┐
   │ h₁  │───▶│ h₂  │───▶│ h₃  │───▶│ h₄  │───▶ output
   └─────┘    └─────┘    └─────┘    └─────┘
```

**Mathematical Formulation:**

The hidden state at time step t is computed as:

$$h_t = \tanh(W_{xh} \cdot x_t + W_{hh} \cdot h_{t-1} + b_h)$$

Where:
- $x_t$ : Input vector at time t
- $h_t$ : Hidden state at time t  
- $W_{xh}$ : Input-to-hidden weight matrix
- $W_{hh}$ : Hidden-to-hidden weight matrix
- $b_h$ : Bias vector

**Output computation:**

$$y_t = W_{hy} \cdot h_t + b_y$$

#### 💡 Key Concepts Reminder: Vanishing Gradient Problem

> **Problem**: During backpropagation through time (BPTT), gradients are multiplied repeatedly by the weight matrix. If eigenvalues < 1, gradients shrink exponentially (vanish); if > 1, they explode. This limits RNN's ability to learn long-range dependencies.
>
> **Mathematical Insight**: For a sequence of length T, the gradient includes terms like:
> $$\frac{\partial h_T}{\partial h_1} = \prod_{t=2}^{T} \frac{\partial h_t}{\partial h_{t-1}}$$

---

#### 2.3.2 Bidirectional RNN (BiRNN)

**Architecture Overview:**

```
Forward:   x₁ ──▶ x₂ ──▶ x₃ ──▶ x₄
            │      │      │      │
            ▼      ▼      ▼      ▼
          [h₁→]  [h₂→]  [h₃→]  [h₄→]
                                  ╲
                                   ╲
                                    ▶ [Concat] ──▶ Output
                                   ╱
                                  ╱
          [h₁←]  [h₂←]  [h₃←]  [h₄←]
            ▲      ▲      ▲      ▲
            │      │      │      │
Backward:  x₁ ◀── x₂ ◀── x₃ ◀── x₄
```

**Mathematical Formulation:**

Forward hidden state:
$$\overrightarrow{h_t} = \tanh(W_{x\overrightarrow{h}} \cdot x_t + W_{\overrightarrow{h}\overrightarrow{h}} \cdot \overrightarrow{h_{t-1}} + b_{\overrightarrow{h}})$$

Backward hidden state:
$$\overleftarrow{h_t} = \tanh(W_{x\overleftarrow{h}} \cdot x_t + W_{\overleftarrow{h}\overleftarrow{h}} \cdot \overleftarrow{h_{t+1}} + b_{\overleftarrow{h}})$$

Combined output:
$$h_t = [\overrightarrow{h_t}; \overleftarrow{h_t}]$$

#### 💡 Key Concepts Reminder: Bidirectional Processing

> **Intuition**: Many NLP tasks benefit from understanding both past and future context. For example, in "The bank of the river was steep," understanding "river" helps disambiguate "bank." BiRNN captures this by processing the sequence in both directions.

---

#### 2.3.3 Gated Recurrent Unit (GRU)

**Architecture Overview:**

```
                    ┌─────────────────────────────────┐
                    │           GRU Cell              │
    x_t ───────────▶│                                 │
                    │   ┌───────────┐                 │
                    │   │ Reset (r) │ ─── Forget Info │
                    │   └───────────┘                 │
                    │   ┌───────────┐                 │
                    │   │Update (z) │ ─── Keep Info   │
                    │   └───────────┘                 │
    h_{t-1} ───────▶│                                 │───────▶ h_t
                    └─────────────────────────────────┘
```

**Mathematical Formulation:**

**Update Gate** (decides how much past information to keep):
$$z_t = \sigma(W_z \cdot [h_{t-1}, x_t] + b_z)$$

**Reset Gate** (decides how much past information to forget):
$$r_t = \sigma(W_r \cdot [h_{t-1}, x_t] + b_r)$$

**Candidate Hidden State:**
$$\tilde{h_t} = \tanh(W_h \cdot [r_t \odot h_{t-1}, x_t] + b_h)$$

**Final Hidden State:**
$$h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h_t}$$

Where:
- $\sigma$ : Sigmoid activation function
- $\odot$ : Element-wise multiplication (Hadamard product)

#### 💡 Key Concepts Reminder: Gating Mechanism

> **Reset Gate (r)**: When r ≈ 0, the candidate hidden state ignores the previous hidden state, allowing the model to "forget" irrelevant information. When r ≈ 1, the full history is considered.
>
> **Update Gate (z)**: Controls the balance between new input and previous memory. When z ≈ 0, the new hidden state mostly copies the old one; when z ≈ 1, it mostly uses the candidate.

---

#### 2.3.4 Long Short-Term Memory (LSTM)

**Architecture Overview:**

```
                ┌────────────────────────────────────────────┐
                │                LSTM Cell                   │
                │                                            │
    C_{t-1} ────│──────────────────────────────────────────▶│──── C_t
                │         ×           +                      │
                │         │     ┌─────┴─────┐                │
                │    ┌────┴──┐  │           │                │
                │    │Forget │  │   ┌───────┴───────┐        │
                │    │ Gate  │  │   │    × tanh    │        │
                │    └───────┘  │   │   ┌───┴───┐  │        │
                │               │   │   │Output │  │        │
                │         ┌─────┴───┴─┐ │ Gate  │  │        │
                │         │   Input   │ └───────┘  │        │
                │         │   Gate    │            │        │
                │         └───────────┘            │        │
                │                                  │        │
    h_{t-1} ────│──────────────────────────────────┴────────│──── h_t
                │                                            │
    x_t ────────│────────────────────────────────────────────│
                └────────────────────────────────────────────┘
```

**Mathematical Formulation:**

**Forget Gate** (what to discard from cell state):
$$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$

**Input Gate** (what new information to store):
$$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$$

**Candidate Cell State:**
$$\tilde{C_t} = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)$$

**Cell State Update:**
$$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C_t}$$

**Output Gate:**
$$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$$

**Hidden State:**
$$h_t = o_t \odot \tanh(C_t)$$

#### 💡 Key Concepts Reminder: Cell State

> **The Cell State (C)**: Acts as a "highway" that allows information to flow unchanged across many time steps. The forget and input gates control additions/removals from this highway, solving the vanishing gradient problem by providing a path for gradients to flow unchanged.
>
> **Key Insight**: LSTM separates memory (C_t) from output (h_t), giving it more expressive power than GRU.

---

#### Models Comparison Summary

| Model | Parameters | Memory Capacity | Training Speed | Best Use Case |
|-------|------------|-----------------|----------------|---------------|
| RNN | Fewest | Limited (short-term) | Fastest | Short sequences |
| BiRNN | 2× RNN | Limited (both directions) | Fast | Context-dependent tasks |
| GRU | Medium | Good | Moderate | Balance speed/accuracy |
| LSTM | Most | Excellent (long-term) | Slowest | Long sequences |
| Bi-LSTM | 2× LSTM | Excellent (bidirectional) | Slowest | Maximum accuracy |

---

### 2.4 Training & Hyperparameter Tuning

#### Loss Function

For regression tasks (predicting relevance scores), Mean Squared Error (MSE) is used:

$$\mathcal{L}_{MSE} = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y_i})^2$$

#### Optimization

**Adam Optimizer** combines momentum and adaptive learning rates:

$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$$
$$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$$
$$\hat{m_t} = \frac{m_t}{1-\beta_1^t}, \quad \hat{v_t} = \frac{v_t}{1-\beta_2^t}$$
$$\theta_t = \theta_{t-1} - \frac{\alpha}{\sqrt{\hat{v_t}} + \epsilon} \hat{m_t}$$

#### Hyperparameter Search Space

| Parameter | Range | Recommended Start |
|-----------|-------|-------------------|
| Hidden Size | [64, 128, 256, 512] | 128 |
| Number of Layers | [1, 2, 3] | 2 |
| Dropout Rate | [0.1, 0.3, 0.5] | 0.3 |
| Learning Rate | [1e-2, 1e-3, 1e-4] | 1e-3 |
| Batch Size | [8, 16, 32, 64] | 16 |
| Bidirectional | [True, False] | True |

#### 💡 Key Concepts Reminder: Regularization

> **Dropout**: During training, randomly sets a fraction p of neurons to zero, preventing co-adaptation. At inference, all neurons are active but outputs are scaled by (1-p).
>
> **Early Stopping**: Monitors validation loss and stops training when it starts increasing, preventing overfitting to the training set.

---

### 2.5 Evaluation Metrics

#### Regression Metrics

**Mean Absolute Error (MAE):**
$$MAE = \frac{1}{N} \sum_{i=1}^{N} |y_i - \hat{y_i}|$$

**Root Mean Squared Error (RMSE):**
$$RMSE = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y_i})^2}$$

**Coefficient of Determination (R²):**
$$R^2 = 1 - \frac{\sum_{i=1}^{N} (y_i - \hat{y_i})^2}{\sum_{i=1}^{N} (y_i - \bar{y})^2}$$

#### BLEU Score (For Text Quality)

BLEU (Bilingual Evaluation Understudy) measures n-gram overlap between generated and reference text:

$$BLEU = BP \cdot \exp\left(\sum_{n=1}^{N} w_n \log p_n\right)$$

Where:
- $p_n$ : Precision of n-grams
- $w_n$ : Weight for n-gram (typically 1/N)
- $BP$ : Brevity penalty for short outputs

#### 💡 Key Concepts Reminder: Metric Selection

> **MAE vs RMSE**: MAE is more robust to outliers (linear penalty), while RMSE penalizes large errors more heavily (quadratic penalty). Choose based on whether large errors are particularly undesirable.
>
> **R² Interpretation**: R² = 0.8 means 80% of variance in the target is explained by the model. Negative R² indicates the model performs worse than predicting the mean.

---

## 3. Part 2: Transformers for Text Generation

### 3.1 GPT-2 Architecture

GPT-2 (Generative Pre-trained Transformer 2) is an autoregressive language model based on the Transformer decoder architecture.

**Architecture Overview:**

```
Input: "The quick brown"
           │
           ▼
┌──────────────────────┐
│   Token Embedding    │ ── Vocabulary: 50,257 tokens
│         +            │
│ Position Embedding   │ ── Max length: 1024 positions
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Transformer Block   │ ×12 (GPT-2 Small)
│  ┌────────────────┐  │
│  │ Masked Self-   │  │
│  │ Attention      │  │
│  └────────────────┘  │
│  ┌────────────────┐  │
│  │ Feed-Forward   │  │
│  │ Network        │  │
│  └────────────────┘  │
│  ┌────────────────┐  │
│  │ Layer Norm     │  │
│  └────────────────┘  │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│   Language Model     │
│   Head (Linear)      │
└──────────┬───────────┘
           │
           ▼
Output: Probability distribution over vocabulary
        → Next token: "fox"
```

#### Self-Attention Mechanism

**Mathematical Formulation:**

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

Where:
- $Q = XW_Q$ : Query matrix
- $K = XW_K$ : Key matrix
- $V = XW_V$ : Value matrix
- $d_k$ : Dimension of keys (scaling factor)

**Multi-Head Attention:**

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W_O$$

Where each head:
$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

#### 💡 Key Concepts Reminder: Attention

> **Intuition**: Attention allows the model to focus on relevant parts of the input when generating each output token. The Query-Key-Value mechanism can be thought of as: "Given what I'm looking for (Query), find the most relevant information (Keys) and retrieve the corresponding content (Values)."
>
> **Scaling Factor**: Division by √d_k prevents dot products from growing too large, which would push softmax into regions with extremely small gradients.

---

### 3.2 Fine-tuning Process

Fine-tuning adapts a pre-trained model to a specific domain or task.

**Process Flow:**

```
┌─────────────────┐
│  Pre-trained    │
│  GPT-2 Model    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Custom Dataset │ ── Domain-specific text
│  Preparation    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Fine-tuning    │ ── Lower learning rate (5e-5)
│  Training       │ ── Fewer epochs (2-4)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Specialized    │
│  Model          │
└─────────────────┘
```

#### Fine-tuning Hyperparameters

| Parameter | Pre-training | Fine-tuning |
|-----------|--------------|-------------|
| Learning Rate | 2.5e-4 | 5e-5 |
| Epochs | Many | 2-4 |
| Batch Size | Large | Small (4-16) |
| Warmup Steps | 2000 | 100-500 |

#### 💡 Key Concepts Reminder: Transfer Learning

> **Why Fine-tune?**: Pre-trained models learn general language understanding from massive datasets. Fine-tuning leverages this knowledge while adapting to specific domains, requiring less data and compute than training from scratch.
>
> **Catastrophic Forgetting**: Using a low learning rate during fine-tuning prevents the model from "forgetting" its pre-trained knowledge while still adapting to new data.

---

### 3.3 Text Generation Techniques

#### Decoding Strategies

**1. Greedy Decoding:**
$$w_t = \arg\max_w P(w|w_{1:t-1})$$

Always selects the highest probability token. Fast but can be repetitive.

**2. Temperature Sampling:**
$$P(w_i) = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$

Where T is temperature:
- T < 1: Sharper distribution (more focused)
- T > 1: Flatter distribution (more random)
- T = 1: Original distribution

**3. Top-k Sampling:**

Restricts sampling to the k most likely tokens:
$$P'(w) = \begin{cases} \frac{P(w)}{\sum_{w \in V_k} P(w)} & \text{if } w \in V_k \\ 0 & \text{otherwise} \end{cases}$$

**4. Top-p (Nucleus) Sampling:**

Samples from smallest set whose cumulative probability exceeds p:
$$V_p = \min\{V' \subseteq V : \sum_{w \in V'} P(w) \geq p\}$$

#### Generation Parameters

| Parameter | Effect | Typical Value |
|-----------|--------|---------------|
| temperature | Randomness control | 0.7 - 1.0 |
| top_k | Vocabulary restriction | 40 - 100 |
| top_p | Dynamic vocabulary | 0.9 - 0.95 |
| max_length | Output length limit | 50 - 200 |
| repetition_penalty | Reduce repetition | 1.1 - 1.3 |

#### 💡 Key Concepts Reminder: Generation Trade-offs

> **Diversity vs. Quality**: Lower temperature/top_k produces more coherent but potentially boring text. Higher values increase creativity but risk generating nonsensical output.
>
> **Top-p vs. Top-k**: Top-k uses a fixed vocabulary size regardless of probability distribution shape. Top-p adapts dynamically—using fewer tokens when the model is confident, more when uncertain.

---

## 4. Mathematical Foundations

### Activation Functions

**Sigmoid:**
$$\sigma(x) = \frac{1}{1 + e^{-x}}$$
- Range: (0, 1)
- Use: Gates in LSTM/GRU

**Tanh:**
$$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$$
- Range: (-1, 1)
- Use: Hidden states, cell state candidates

**ReLU:**
$$\text{ReLU}(x) = \max(0, x)$$
- Range: [0, ∞)
- Use: Feed-forward layers

### Backpropagation Through Time (BPTT)

For RNNs, gradients are computed by unrolling the network through time:

$$\frac{\partial \mathcal{L}}{\partial W} = \sum_{t=1}^{T} \frac{\partial \mathcal{L}_t}{\partial W}$$

$$\frac{\partial \mathcal{L}_t}{\partial W_{hh}} = \sum_{k=1}^{t} \frac{\partial \mathcal{L}_t}{\partial h_t} \frac{\partial h_t}{\partial h_k} \frac{\partial h_k}{\partial W_{hh}}$$

### Cross-Entropy Loss (for Language Modeling)

$$\mathcal{L}_{CE} = -\sum_{i=1}^{V} y_i \log(\hat{y_i})$$

For next-token prediction:
$$\mathcal{L} = -\log P(w_t | w_{1:t-1})$$

---

## 5. Key Concepts Reminder

### Quick Reference Card

| Concept | Definition | Importance |
|---------|------------|------------|
| **Embedding** | Dense vector representation of tokens | Captures semantic meaning |
| **Hidden State** | Internal representation at each time step | Encodes sequence history |
| **Cell State** | Long-term memory in LSTM | Enables learning long dependencies |
| **Attention** | Mechanism to focus on relevant inputs | Foundation of Transformers |
| **Fine-tuning** | Adapting pre-trained models | Efficient transfer learning |
| **Dropout** | Random neuron deactivation | Prevents overfitting |
| **Gradient Clipping** | Limiting gradient magnitude | Prevents exploding gradients |

### Common Pitfalls & Solutions

| Problem | Symptom | Solution |
|---------|---------|----------|
| Vanishing Gradients | Training stalls, no learning | Use LSTM/GRU, gradient clipping |
| Overfitting | Low train loss, high val loss | Add dropout, early stopping |
| Underfitting | High train and val loss | Increase model capacity |
| Exploding Gradients | NaN losses | Gradient clipping, lower LR |
| Mode Collapse (Generation) | Repetitive outputs | Temperature sampling, top-p |

---

## 6. Practical Implementation Guide

### Environment Setup

```bash
# Required packages
pip install torch transformers scikit-learn pandas numpy matplotlib

# For Arabic NLP
pip install arabert farasapy
# Or install from GitHub if pip fails:
pip install git+https://github.com/aub-mind/arabert.git
```

### Training Pipeline Summary

```python
# 1. Data Preparation
dataset = preprocess(raw_data)
train_loader, val_loader = create_dataloaders(dataset)

# 2. Model Definition
model = SequenceModel(vocab_size, hidden_size, num_layers)

# 3. Training Loop
for epoch in range(num_epochs):
    for batch in train_loader:
        optimizer.zero_grad()
        outputs = model(batch)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
    
    # Validation
    val_loss, val_metrics = evaluate(model, val_loader)
    
    # Early stopping check
    if val_loss < best_loss:
        save_checkpoint(model)

# 4. Evaluation
test_metrics = evaluate(model, test_loader)
```

### GPU Acceleration

```python
# Check GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Move model and data to GPU
model = model.to(device)
inputs = inputs.to(device)
```

---

## 7. Conclusion

This lab provided comprehensive hands-on experience with:

1. **Data Collection & Preprocessing**: Building a complete pipeline for Arabic NLP, including web scraping and text normalization specific to Arabic language characteristics.

2. **Sequence Models**: Deep understanding of RNN architectures (vanilla RNN, BiRNN, GRU, LSTM), their mathematical foundations, and practical trade-offs between complexity and performance.

3. **Training & Evaluation**: Implementing proper training procedures with hyperparameter tuning, regularization techniques, and comprehensive evaluation using appropriate metrics.

4. **Transformers**: Understanding the attention mechanism and GPT-2 architecture, along with practical fine-tuning and text generation techniques.

### Key Takeaways

- **Architecture Selection**: Start with simpler models (GRU) and increase complexity only if needed
- **Data Quality**: Preprocessing quality significantly impacts model performance
- **Hyperparameter Tuning**: Systematic exploration is essential for optimal results
- **Transfer Learning**: Pre-trained models dramatically reduce training requirements
- **Evaluation**: Choose metrics appropriate to the task and business requirements

---

## 8. References

1. Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory. Neural Computation.
2. Cho, K., et al. (2014). Learning Phrase Representations using RNN Encoder-Decoder.
3. Vaswani, A., et al. (2017). Attention Is All You Need.
4. Radford, A., et al. (2019). Language Models are Unsupervised Multitask Learners (GPT-2).
5. Antoun, W., et al. (2020). AraBERT: Transformer-based Model for Arabic Language Understanding.

---

**Document Version:** 1.0  
**Last Updated:** December 2024  
**Author:** Master MBD Student  
**Course:** Deep Learning - Lab 3
