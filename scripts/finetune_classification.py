import argparse

import evaluate
import numpy as np
import torch
from toddbenchmark.classification_datasets import prep_dataset, prep_model
from toddbenchmark.classification_datasets_configs import DATASETS_CONFIGS
from transformers import (
    Trainer,
    TrainingArguments,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Finetune a classification model on a dataset"
    )
    parser.add_argument("--model_name", type=str, default="distilbert-base-uncased")
    parser.add_argument(
        "--dataset_config",
        type=str,
        help="Not huggingface dataset config but config presented in this repo",
        default="sst2",
        choices=list(DATASETS_CONFIGS.keys()),
    )

    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--max_length", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--output_file", type=str, default="output.txt")
    return parser.parse_args()


accuracy = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)


if __name__ == "__main__":
    args = parse_args()

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epoch,
        save_steps=1000,
        logging_strategy="steps",
        logging_steps=1000,
        seed=args.seed,
    )

    model, tokenizer = prep_model(
        args.model_name, DATASETS_CONFIGS[args.dataset_config]
    )
    train_dataset, validation_dataset, _ = prep_dataset(
        args.dataset_config, DATASETS_CONFIGS[args.dataset_config], tokenizer
    )

    def tokenize_function(examples):
        return tokenizer(text=examples["text"], truncation=True)

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

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()
