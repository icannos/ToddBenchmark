import argparse

import numpy as np
import torch
from evaluate import load as load_metric
from transformers import (
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)

from toddbenchmark.generation_datasets import prep_dataset, prep_model
from toddbenchmark.generation_datasets_configs import DATASETS_CONFIGS


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a model on a dataset")
    parser.add_argument("--model_name", type=str, default="Helsinki-NLP/opus-mt-de-en")
    parser.add_argument(
        "--dataset_config",
        type=str,
        default="wmt16_de_en",
        choices=list(DATASETS_CONFIGS.keys()),
    )
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--max_length", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--output_dir", type=str, default="output_models")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    metrics = {
        "Accuracy": load_metric("accuracy"),
        "BLEU": load_metric("sacrebleu"),
        "rouge": load_metric("rouge"),
    }

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)

        return {
            k: m.compute(predictions=predictions, references=labels)
            for k, m in metrics.items()
        }

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epoch,
        save_steps=1000,
        logging_strategy="steps",
        logging_steps=1000,
    )

    model, tokenizer = prep_model(args.model_name)
    config = DATASETS_CONFIGS[args.dataset_config]
    train_dataset, validation_dataset, _ = prep_dataset(
        config["dataset_name"],
        config["dataset_config"],
        tokenizer,
        train_max_size=200,
        validation_max_size=200,
        test_max_size=200,
    )

    def tokenize_function(examples):
        inputs = tokenizer(examples["source"], truncation=True, padding="max_length")
        targets = tokenizer(examples["target"], truncation=True, padding="max_length")
        inputs["labels"] = targets["input_ids"]
        return inputs

    train_dataset = train_dataset.map(
        tokenize_function,
        batched=True,
        num_proc=4,
    )
    validation_dataset = validation_dataset.map(
        tokenize_function,
        batched=True,
        num_proc=4,
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=tokenizer.pad_token_id,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
    )

    trainer.train()
