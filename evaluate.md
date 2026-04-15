# Evaluating LLMs with Hugging Face Evaluate Library

Final section from DataCamp LLM course - **how to properly evaluate** your language models!

## Why Special Metrics for LLMs?
- Classic ML metrics (accuracy) work, **but language tasks need more**
- **Hugging Face Evaluate library** = 50+ metrics for NLP tasks [web:1]
- Covers: classification, generation, summarization, translation, Q&A

## How Evaluate Works
```python
import evaluate

# Load any metric
accuracy = evaluate.load("accuracy")
print(accuracy.description)  # Metric explanation
print(accuracy.features)     # Required inputs (predictions, references)
```

**Most metrics need**:
- `predictions`: Your model outputs
- `references`: Ground truth labels

## Classification Example
```python
# Load multiple metrics
accuracy = evaluate.load("accuracy")
precision = evaluate.load("precision")
f1 = evaluate.load("f1")
recall = evaluate.load("recall")

# Fake predictions vs real labels
predictions =   # Model guesses[1]
references =    # True labels[1]

# Compute scores
print(accuracy.compute(predictions=predictions, references=references))
# {'accuracy': 0.8}
```

**Results explained**:
- 80% correct overall
- Perfect precision (no false positives)
- Some false negatives (missed positives)

## Real Fine-tuned Model Test
```python
# Load your fine-tuned model
model = AutoModelForSequenceClassification.from_pretrained("your-model")
tokenizer = AutoTokenizer.from_pretrained("your-model")

# Test on new data
test_texts = ["Love this!", "Terrible product"]
inputs = tokenizer(test_texts, return_tensors="pt", padding=True)
outputs = model(**inputs)
predictions = outputs.logits.argmax(-1).tolist()

# Evaluate
results = accuracy.compute(predictions=predictions, references=)[1]
```

## ⚠️ Metric Choice Matters!
| Scenario | Don't Use | Use Instead |
|----------|-----------|-------------|
| Imbalanced data | Accuracy alone | F1 + Precision + Recall |
| Perfect scores | Trust blindly | Check for data leakage |
| Business use | Generic metrics | Domain KPIs too |

**Golden Rule**: **Combine multiple metrics** for real understanding!

**Takeaway**: Evaluate library makes LLM testing easy + comprehensive. Perfect scores? Double-check your data!