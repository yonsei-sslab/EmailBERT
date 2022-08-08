import multiprocessing
import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    TrainingArguments,
    Trainer,
)
from utils import (
    seed_everything,
    load_config,
    tokenize_function,
    group_texts,
    whole_word_masking_data_collator,
)

CFG = load_config()
seed_everything(CFG.seed)
tokenizer = AutoTokenizer.from_pretrained(CFG.model_checkpoint)
model = AutoModelForMaskedLM.from_pretrained(CFG.model_checkpoint)
dset = load_dataset(CFG.data_path)


# Use batched=True to activate fast multithreading!
tokenized_datasets = dset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"],
    num_proc=multiprocessing.cpu_count() // 2,
)

# Slicing produces a list of lists for each feature
tokenized_samples = tokenized_datasets["train"][:3]
for idx, sample in enumerate(tokenized_samples["input_ids"]):
    print(f"'>>> email {idx} length: {len(sample)}'")

concatenated_examples = {k: sum(tokenized_samples[k], []) for k in tokenized_samples.keys()}
total_length = len(concatenated_examples["input_ids"])
print(f"'>>> Concatenated email length: {total_length}'")

chunks = {
    k: [t[i : i + CFG.chunk_size] for i in range(0, total_length, CFG.chunk_size)]
    for k, t in concatenated_examples.items()
}

for chunk in chunks["input_ids"]:
    print(f"'>>> Chunk length: {len(chunk)}'")


lm_datasets = tokenized_datasets.map(group_texts, batched=True, num_proc=6)
print(tokenizer.decode(lm_datasets["train"][1]["input_ids"]))
print(tokenizer.decode(lm_datasets["train"][1]["labels"]))

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm_probability=CFG.masking_probability
)

samples = [lm_datasets["train"][i] for i in range(2)]
for sample in samples:
    _ = sample.pop("word_ids")

for chunk in data_collator(samples)["input_ids"]:
    print(f"\n'>>> {tokenizer.decode(chunk)}'")


samples = [lm_datasets["train"][i] for i in range(2)]
batch = whole_word_masking_data_collator(samples)

for chunk in batch["input_ids"]:
    print(f"\n'>>> {tokenizer.decode(chunk)}'")

downsampled_dataset = lm_datasets["train"].train_test_split(
    train_size=CFG.train_size, test_size=CFG.test_size, seed=CFG.seed
)
print(downsampled_dataset)
# Show the training loss with every epoch
epoch_step = len(downsampled_dataset["train"]) // (CFG.batch_size * torch.cuda.device_count())
model_name = CFG.model_checkpoint.split("/")[-1]

if CFG.early_stop <= -1:
    # no early stopping
    training_args = TrainingArguments(
        output_dir=f"{model_name}-email-finetuned",
        overwrite_output_dir=True,
        learning_rate=CFG.learning_rate,
        weight_decay=CFG.weight_decay,
        warmup_ratio=CFG.warm_up_ratio,
        per_device_train_batch_size=CFG.batch_size,
        per_device_eval_batch_size=CFG.batch_size,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=CFG.num_train_epochs,
        do_eval=True,
        push_to_hub=True,
        fp16=CFG.fp16,
        gradient_checkpointing=CFG.gradient_checkpointing,
        logging_steps=CFG.logging_steps,
        save_total_limit=CFG.save_total_limit,
        metric_for_best_model="loss",
        load_best_model_at_end=True,
    )
elif CFG.early_stop > 0:
    training_args = TrainingArguments(
        output_dir=f"{model_name}-email-finetuned",
        overwrite_output_dir=True,
        learning_rate=CFG.learning_rate,
        weight_decay=CFG.weight_decay,
        warmup_ratio=CFG.warm_up_ratio,
        per_device_train_batch_size=CFG.batch_size,
        per_device_eval_batch_size=CFG.batch_size,
        evaluation_strategy="steps",
        eval_steps=epoch_step,
        save_steps=epoch_step,
        num_train_epochs=CFG.num_train_epochs,
        do_eval=True,
        push_to_hub=True,
        fp16=CFG.fp16,
        gradient_checkpointing=CFG.gradient_checkpointing,
        logging_steps=CFG.logging_steps,
        save_total_limit=CFG.save_total_limit,
        metric_for_best_model="loss",
        load_best_model_at_end=True,
    )

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=downsampled_dataset["train"],
    eval_dataset=downsampled_dataset["test"],
    data_collator=data_collator,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=CFG.early_stop)],
)

trainer.train()
trainer.push_to_hub()
