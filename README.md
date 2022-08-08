# EmailBERT

Pretrained RoBERTa finetuned on Enron email dataset & aeslc email dataset with Masked Language Modeling train objective.

### How to use

```python
from transformers import pipeline

mask_filler = pipeline(
    "fill-mask", model="snoop2head/EmailBERT-base"
)
text = f"nPlease cc the following distribution list with updates:\nPhillip Allen (pallen@enron.com)\nMike Grigsby (mike.grigsby@enron.com)\nKeith Holst (kholst@enron.com)\nMonique Sanchez\n {tokenizer.mask_token} Ermis\nJohn Lavorato\nThank you for your help\nPhillip Allen'"

preds = mask_filler(text)

for pred in preds:
    print(f"Mask Filled: {pred['sequence']}")
```

### References

- Dataset: https://huggingface.co/datasets/snoop2head/enron_aeslc_emails
- MLM Finetuning: https://huggingface.co/course/chapter7/3?fw=pt
