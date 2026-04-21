# 🔧 Preparing for Fine-Tuning LLMs

## 👋 Introduction
So far, we've used the `pipeline()` interface in Hugging Face, which is **easy to use** but **limits customization**. To fine-tune models, we need more control, which we get using **auto classes**.

---

## ⚙️ Pipelines vs Auto Classes

| Feature            | Pipeline          | Auto Classes             |
|------------------|-----------------|------------------------|
| Ease of use        | ✅ Simple        | ⚠️ Requires setup       |
| Customization      | ❌ Limited       | ✅ Full control         |
| Fine-tuning        | ❌ Not flexible | ✅ Supports fine-tuning |

---

## 🧠 LLM Development Lifecycle

LLMs are trained in **two main phases**:

1. **Pre-training**  
   - Trained on large general datasets  
   - Learns general language patterns

2. **Fine-tuning**  
   - Trained on domain-specific data  
   - Adapts model for specialized tasks  
   - Example: Fine-tuning on insurance data to answer customer queries

---

## 📥 Loading a Dataset for Fine-Tuning

- Use Hugging Face’s **`datasets` library** to access pre-made datasets.
- Example: Using **IMDB movie reviews** dataset.
- Steps:
  1. Load dataset with `load_dataset("imdb", split="train")`
  2. Use `.shard()` to split the dataset into chunks (helps speed up training)
  3. Select a chunk for experimentation

---

## 🏗️ Using Auto Classes

- **AutoModel** → Loads a model architecture
- **AutoTokenizer** → Loads the tokenizer for that model
- **Task-specific classes**:
  - `AutoModelForSequenceClassification` → for sentiment classification
- Use `from_pretrained()` to load a pre-trained model with its weights and tokenizer

---

## 📝 Tokenization

- Converts text into **token IDs** the model can understand
- Example settings:
  - Enable **padding** for shorter sequences
  - Enable **truncation** for sequences longer than max length
  - Use `return_tensors="pt"` for PyTorch

### ✅ Example Output
- A list of **token IDs**
- Can tokenize:
  - Entire dataset at once
  - Row by row using `.map()` for custom control

---

## 🔹 Subword Tokenization

- Words are broken into smaller **subword units**
- Helps handle **rare words** efficiently
- Example:
  - `"unbelievably"` → `"un"`, `"believ"`, `"ably"`

---

## 🧾 Summary (Simple Explanation)

Fine-tuning involves preparing a dataset, loading a pre-trained model and tokenizer using Hugging Face **auto classes**, and tokenizing your data (often using subword tokenization). This allows the model to **adapt to domain-specific tasks** while building on the knowledge it learned during pre-training.

---