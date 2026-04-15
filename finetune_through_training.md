# 🏋️ Fine-Tuning LLMs Through Training

## 👋 Introduction
Once the dataset is prepared and tokenized, the next step is to **fine-tune the model** using Hugging Face’s **Trainer API**.

---

## ⚙️ Training Arguments

The **`TrainingArguments`** class helps control **how training happens**:

| Parameter                     | Purpose                                                                 |
|--------------------------------|------------------------------------------------------------------------|
| `output_dir`                   | Where to save the model predictions                                     |
| `evaluation_strategy`          | When to evaluate model (e.g., `"epoch"` evaluates after each epoch)    |
| `num_train_epochs`             | Number of times the model will see the full training data               |
| `learning_rate`                | Step size for optimizer updates                                         |
| `per_device_train_batch_size`  | Batch size during training                                              |
| `per_device_eval_batch_size`   | Batch size during evaluation                                           |
| `weight_decay`                 | Helps avoid overfitting, especially on small datasets                  |

> Tip: Values depend on dataset size, task, and available compute.

---

## 🏗️ Trainer Class

- Wrap your **model**, **training arguments**, **datasets**, and **tokenizer** into a **`Trainer`** object.
- Training is started using:

```python
trainer.train()