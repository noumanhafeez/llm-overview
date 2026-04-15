# Using Pre-trained LLMs: Language Tasks

Continuation from DataCamp LLM course - practical **Hugging Face pipelines** for language work!

## Language Tasks LLMs Handle
- **Understanding**: Text classification, sentiment analysis, summarization, Q&A
- **Generation**: Text creation + translation (focus here)

## 1. Text Generation
**Goal**: Extend prompts into **human-like text**

```python
from transformers import pipeline

generator = pipeline("text-generation", model="gpt2")

prompt = "Traditional Japanese houses are built with..."
result = generator(
    prompt, 
    max_length=50,
    pad_token_id=generator.tokenizer.eos_token_id,
    truncation=True
)

print(result['generated_text'])
```

### Key Parameters
| Param | What it does |
|-------|-------------|
| `max_length` | Caps total output length |
| `pad_token_id` | Padding + end-of-text marker |
| `truncation=True` | Trims long inputs |

**Pro Tip**: Specific prompts = better results!
❌ Vague: "Houses are cool" → Random output
✅ Guided: f"Review: 'Great!' Response: I agree because..."


## 2. Language Translation
**Goal**: Convert text while **keeping meaning**

```python
translator = pipeline("translation_en_to_es")

text = "Traditional Japanese houses are built with wood."
result = translator(text, clean_up_tokenization_spaces=True)

print(result['translation_text'])
```

## Output Keys Quick Reference
| Task | Access Key |
|------|------------|
| Text Generation | `result[0]['generated_text']` |
| Translation | `result[0]['translation_text']` |
| Summarization | `result[0]['summary_text']` |

**Key Takeaway**: Pipelines = instant LLM power. Good prompts = great results!