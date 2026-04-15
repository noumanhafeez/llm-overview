## LLM Evaluation Metrics: Perplexity & BLEU

When evaluating language models (LLMs), classical metrics like accuracy or F1 score are often not enough. For text generation, summarization, or translation tasks, **specialized metrics** like **Perplexity** and **BLEU** are commonly used.

---

### 1. Perplexity

- **Purpose:** Measures how confident the model is in predicting the next word in a sequence.
- **Meaning:** Lower perplexity → higher confidence → better predictions.
- **How it works:**
  1. Input text is converted to token IDs using `.encode()`.
  2. Model generates token IDs for the output using `.generate()`.
  3. Output token IDs are converted back to human-readable text with `.decode()`.
- **Metric Calculation:** The probabilities assigned by the model to each predicted word are used to calculate perplexity.
- **Interpretation:** Compare `mean_perplexity` against a baseline to understand model performance.

---

### 2. BLEU (Bilingual Evaluation Understudy)

- **Purpose:** Measures quality of generated text by comparing it to human-provided references.
- **Use Cases:** Text generation, translation, summarization.
- **How it works:**
  1. Store model predictions and human references in variables.
  2. Pass them to the BLEU metric’s `.compute()` method.
- **Output:** Score between 0 and 1.
  - **1 → Perfect match** with the reference.
  - **Closer to 1 → Higher similarity**.

---

### Summary

- **Perplexity:** Evaluates **confidence** of predictions. Lower is better.  
- **BLEU:** Evaluates **similarity to human references**. Higher is better.  

These metrics are essential for understanding how well a language model performs in generating accurate and human-like text.