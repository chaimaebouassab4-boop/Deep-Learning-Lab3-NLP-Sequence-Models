# Deep-Learning-Lab3-NLP-Sequence-Models
Dive into PyTorch-powered NLP adventures! This repo showcases Lab 3 from the MBD Deep Learning course: scraping Arabic texts, preprocessing pipelines, training RNN, BiRNN, GRU, and LSTM models for classification, plus fine-tuning GPT-2 for dynamic text generation. 
# Arabic NLP Lab: Sequence Models & Text Generation

## 📋 Overview

This lab explores Natural Language Processing (NLP) techniques for Arabic text, progressing from classical sequence models (RNN, GRU, LSTM) to modern Transformers (GPT-2). The lab consists of two main parts:

1. **Text Classification**: Build and compare sequence models for relevance scoring
2. **Text Generation**: Fine-tune GPT-2 for Arabic paragraph generation

**Domain Focus**: Technology news articles (Arabic)

---

## 🎯 Learning Objectives

By completing this lab, we will:

- Understand the evolution from RNNs to Transformers in NLP
- Master Arabic text preprocessing pipelines
- Implement and compare 4 sequence model architectures
- Fine-tune pre-trained language models
- Evaluate models using standard metrics and BLEU scores
- Handle morphologically rich languages in deep learning

---

## 🏗️ Lab Structure

```
arabic-nlp-lab/
├── README.md                    # This file
├── lab3_notebook.ipynb          # Main Jupyter notebook
├── data/
│   ├── raw/                     # Scraped Arabic texts
│   └── processed/               # Cleaned and tokenized data
├── models/
│   ├── rnn_model.pth
│   ├── birnn_model.pth
│   ├── gru_model.pth
│   ├── lstm_model.pth
│   └── gpt2_finetuned/          # Fine-tuned Transformer
├── results/
│   └── evaluation_metrics.csv
└── requirements.txt
```

---

## 📚 Key Concepts

### **NLP Fundamentals**
- **Natural Language Processing**: AI field enabling computers to understand and generate human language
- **Challenges**: Context ambiguity, Arabic's root-based morphology, diacritics

### **Sequence Models** (Process text where order matters)

| Model | Architecture | Strengths | Weaknesses |
|-------|-------------|-----------|------------|
| **RNN** | Unidirectional hidden state | Simple, captures order | Vanishing gradients |
| **Bidirectional RNN** | Forward + backward passes | Better context | 2x computation |
| **GRU** | Update & reset gates | Handles long dependencies | Less expressive than LSTM |
| **LSTM** | Cell state + 3 gates | Best for long sequences | Most parameters |

### **Transformers**
- **Architecture**: Self-attention mechanism (parallel processing)
- **GPT-2**: Pre-trained generative model for text completion
- **Advantages**: No sequential bottleneck, captures long-range dependencies

### **Evaluation Metrics**
- **Classification**: Accuracy, Precision, Recall, F1-Score
- **Generation**: BLEU Score (n-gram overlap with references)

---

## 🚀 Part 1: Classification Task

### **Objective**
Classify Arabic news articles by relevance score (0-10) using sequence models.

### **Workflow**

#### 1. Data Collection
```python
# Scrape Arabic technology news
from bs4 import BeautifulSoup
import requests

# Example sources:
# - Al Jazeera Tech: aljazeera.net/technology
# - Arabic Wikipedia tech articles
# - Tech blogs in Arabic

# Output: CSV with columns ["Text", "Score"]
```

**Requirements**:
- Minimum 200 articles
- Manual relevance scoring (0-10 scale)
- UTF-8 encoding for Arabic
- Check `robots.txt` for ethical scraping

#### 2. Preprocessing Pipeline
```python
# Arabic-specific preprocessing
from camel_tools.tokenizers.word import simple_word_tokenize
from camel_tools.morphology.database import MorphologyDB
from camel_tools.morphology.analyzer import Analyzer

steps = [
    "Remove diacritics (تشكيل)",
    "Tokenize using CAMeL Tools",
    "Remove stop words (حروف الجر)",
    "Lemmatization (Arabic roots)",
    "Discretize scores → classes",
    "Convert to numerical sequences"
]
```

**Example**:
```
Input:  "التكنولوجيا الحديثة تُغيّر العالم" (Score: 8)
        ↓
Output: [45, 823, 12, 934, 2] (Class: Relevant)
```

#### 3. Model Implementation

**Architecture Template** (PyTorch):
```python
import torch.nn as nn

class ArabicRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        embedded = self.embedding(x)
        output, hidden = self.rnn(embedded)
        return self.fc(hidden[-1])
```

**Models to Implement**:
1. Vanilla RNN
2. Bidirectional RNN
3. GRU (Gated Recurrent Unit)
4. LSTM (Long Short-Term Memory)

#### 4. Hyperparameter Tuning
```python
hyperparams = {
    'learning_rate': [1e-3, 1e-4, 1e-5],
    'batch_size': [16, 32, 64],
    'hidden_dim': [128, 256, 512],
    'epochs': [10, 20, 30]
}

# Use Optuna or GridSearchCV for optimization
```

#### 5. Evaluation
```python
from sklearn.metrics import classification_report, f1_score

# Compute metrics on test set
results = {
    'RNN': {'accuracy': 0.72, 'f1': 0.70},
    'BiRNN': {'accuracy': 0.78, 'f1': 0.76},
    'GRU': {'accuracy': 0.82, 'f1': 0.80},
    'LSTM': {'accuracy': 0.85, 'f1': 0.83}  # Best
}
```

**Expected Findings**:
- LSTM/GRU outperform vanilla RNN (better gradient flow)
- Bidirectional models improve context understanding
- Arabic morphology benefits from gating mechanisms

---

## 🤖 Part 2: Text Generation with Transformers

### **Objective**
Fine-tune GPT-2 on Arabic technology corpus for coherent paragraph generation.

### **Workflow**

#### 1. Setup
```bash
pip install transformers torch datasets
```

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer

# Load pre-trained model (or Arabic variant like AraGPT2)
model_name = "aubmindlab/aragpt2-base"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
```

#### 2. Dataset Preparation
```python
# Create training corpus
texts = [
    "الذكاء الاصطناعي يُحدث ثورة في مجال التكنولوجيا...",
    "الحوسبة السحابية تُمكّن الشركات من تحسين أدائها...",
    # ... 100+ paragraphs
]

# Tokenize and format
train_encodings = tokenizer(texts, truncation=True, padding=True)
```

#### 3. Fine-Tuning
```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir='./gpt2_finetuned',
    num_train_epochs=3,
    per_device_train_batch_size=4,
    learning_rate=5e-5,
    warmup_steps=500,
    save_steps=1000
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset
)

trainer.train()
```

**Key Parameters**:
- Small learning rate (5e-5) to avoid catastrophic forgetting
- Few epochs (3-5) for specialization
- Gradient accumulation for limited GPU memory

#### 4. Text Generation
```python
prompt = "الذكاء الاصطناعي في المستقبل"
input_ids = tokenizer.encode(prompt, return_tensors='pt')

output = model.generate(
    input_ids,
    max_length=150,
    num_return_sequences=1,
    temperature=0.8,  # Controls randomness
    top_k=50,
    top_p=0.95,
    do_sample=True
)

generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)
```

**Generation Parameters**:
- `temperature`: 0.7-1.0 (higher = more creative)
- `top_k`: Limits vocabulary per step (50 works well)
- `top_p`: Nucleus sampling threshold (0.9-0.95)

#### 5. Evaluation
```python
from nltk.translate.bleu_score import sentence_bleu

# Compare generated text with human references
reference = [["الذكاء", "الاصطناعي", "سيغير", "العالم"]]
candidate = ["الذكاء", "الاصطناعي", "يحدث", "ثورة"]

bleu_score = sentence_bleu(reference, candidate)
print(f"BLEU Score: {bleu_score:.4f}")
```

---

## 🛠️ Installation & Setup

### **Prerequisites**
- Python 3.8+
- CUDA-enabled GPU (recommended for Transformers)
- 8GB+ RAM

### **Environment Setup**
```bash
# Clone repository
git clone https://github.com/yourusername/arabic-nlp-lab.git
cd arabic-nlp-lab

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### **requirements.txt**
```
torch>=2.0.0
transformers>=4.35.0
camel-tools>=1.5.2
beautifulsoup4>=4.12.0
scrapy>=2.11.0
optuna>=3.4.0
nltk>=3.8.1
scikit-learn>=1.3.0
pandas>=2.1.0
matplotlib>=3.8.0
seaborn>=0.13.0
jupyter>=1.0.0
```

---

## 📊 Running the Lab

### **Option 1: Google Colab (Recommended)**
```python
# In Colab notebook:
!git clone https://github.com/yourusername/arabic-nlp-lab.git
%cd arabic-nlp-lab
!pip install -r requirements.txt

# Open lab3_notebook.ipynb
# Runtime → Change runtime type → GPU (T4)
```

### **Option 2: Kaggle**
```bash
# Upload notebook to Kaggle
# Settings → Accelerator → GPU
# Add dataset from "data/raw/"
```

### **Option 3: Local Jupyter**
```bash
jupyter notebook lab3_notebook.ipynb
```

### **Execution Steps**
1. **Cell 1-3**: Import libraries and check environment
2. **Cell 4-8**: Data collection and scraping
3. **Cell 9-15**: Preprocessing pipeline
4. **Cell 16-30**: Model training (RNN, BiRNN, GRU, LSTM)
5. **Cell 31-35**: Evaluation and comparison
6. **Cell 36-45**: Transformer fine-tuning
7. **Cell 46-50**: Text generation experiments

---

## 📈 Expected Results

### **Part 1: Classification**
| Model | Accuracy | F1-Score | Training Time |
|-------|----------|----------|---------------|
| RNN | 72-75% | 0.70-0.73 | ~5 min |
| BiRNN | 76-80% | 0.74-0.78 | ~8 min |
| GRU | 80-84% | 0.78-0.82 | ~6 min |
| **LSTM** | **83-87%** | **0.81-0.85** | ~10 min |

**Key Insight**: LSTM's cell state mechanism best handles Arabic's complex morphology.

### **Part 2: Generation**
```
Input: "الذكاء الاصطناعي في المستقبل"

Output (Fine-tuned GPT-2):
"الذكاء الاصطناعي في المستقبل سيلعب دورًا محوريًا في تحويل 
الصناعات التقليدية. من المتوقع أن تشهد تقنيات التعلم العميق 
تطورًا كبيرًا يمكّن الآلات من فهم السياق اللغوي بدقة أعلى..."

BLEU Score: 0.62 (Good coherence)
```

---

## 🐛 Troubleshooting

### **Common Issues**

#### 1. Arabic Text Encoding Errors
```python
# Solution: Force UTF-8 encoding
with open('data.csv', 'r', encoding='utf-8-sig') as f:
    data = f.read()
```

#### 2. Out of Memory (GPU)
```python
# Solution: Reduce batch size
batch_size = 8  # Instead of 32
# Or use gradient accumulation
accumulation_steps = 4
```

#### 3. CAMeL Tools Installation
```bash
# Solution: Install from source
pip install camel-tools --no-cache-dir
python -m spacy download en_core_web_sm
```

#### 4. Vanishing Gradients in RNN
```python
# Solution: Use gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

---

## 🧪 Experimental Extensions

### **Advanced Challenges**
1. **Multi-label Classification**: Predict multiple topics per article
2. **Cross-lingual Transfer**: Train on Arabic, test on English
3. **Custom Tokenizer**: Build BPE tokenizer for Arabic corpus
4. **Model Compression**: Apply quantization to reduce model size
5. **Zero-shot Learning**: Use mBERT for unseen categories

### **Research Questions**
- How does diacritic removal affect model accuracy?
- Can attention mechanisms improve RNN performance?
- Does pre-training on MSA help with dialectal Arabic?

---

## 📖 Additional Resources

### **Arabic NLP**
- [CAMeL Tools Documentation](https://camel-tools.readthedocs.io/)
- [AraVec Word Embeddings](https://github.com/bakrianoo/aravec)
- [Arabic BERT Models](https://huggingface.co/aubmindlab)

### **Deep Learning**
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [Dive into Deep Learning](https://d2l.ai/)
- [Understanding LSTMs](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)

### **Transformers**
- [Hugging Face Course](https://huggingface.co/course)
- [Annotated Transformer](http://nlp.seas.harvard.edu/annotated-transformer/)
- [GPT-2 Paper](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)

### **Tutorial Reference**
- [RNN Text Generation Guide](https://gist.github.com/mf1024/3df214d2f17f3dcc56450ddf0d5a4cd7)

---

## 🎓 Learning Synthesis

**After completing this lab, you should understand:**

1. **Arabic NLP Challenges**: Morphological richness requires specialized preprocessing (stemming, diacritic handling)

2. **Sequence Model Evolution**:
   - RNNs struggle with long-term dependencies (vanishing gradients)
   - GRU/LSTM gates enable better information flow
   - Bidirectional processing improves context capture

3. **Transformer Advantages**:
   - Parallel processing (vs. sequential RNNs)
   - Self-attention captures global dependencies
   - Transfer learning via pre-training

4. **Practical Insights**:
   - Fine-tuning requires careful learning rate selection
   - BLEU scores correlate with human fluency judgments
   - Data quality matters more than model complexity

---

## 📝 Submission Checklist

- [ ] Completed `lab3_notebook.ipynb` with all cells executed
- [ ] Scraped dataset saved in `data/raw/`
- [ ] Trained models saved in `models/`
- [ ] Evaluation metrics exported to `results/`
- [ ] README.md updated with findings
- [ ] Synthesis paragraph written
- [ ] Code committed to GitHub with meaningful messages
- [ ] Final notebook exported as PDF

---

## 👥 Contributors

- **Your Name** - Implementation and experiments
- **Course Instructor** - Lab design and guidance

---

## 📅 Version History

- **v1.0** (2025-01-XX): Initial lab release
- **v1.1** (2025-01-XX): Added Transformer section
- **v1.2** (2025-01-XX): Completed with results

---

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- CAMeL Lab at NYU Abu Dhabi for Arabic NLP tools
- Hugging Face for Transformers library
- PyTorch team for deep learning framework
- Arabic NLP research community

---

**Last Updated**: January 2025  
**Lab Duration**: 8-12 hours  
**Difficulty**: Intermediate to Advanced

**Happy Learning! 🚀📚**
