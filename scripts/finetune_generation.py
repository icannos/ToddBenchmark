import nltk

print("TEST")

import argparse
from pathlib import Path

import numpy as np
import torch
from evaluate import load as load_metric
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    IntervalStrategy
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
        "BLEU": load_metric("sacrebleu"),
        "rouge": load_metric("rouge"),
    }


    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]

        # rougeLSum expects newline after each sentence
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

        return preds, labels

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        result = {}

        for metric_name, metric in metrics.items():
            result = result | {f"{metric_name}: {k}": v for k, v in metric.compute(predictions=decoded_preds, references=decoded_labels).items()}
        return result

    # create output dir if not exists
    print("HEY!!!")
    Path(args.output_dir).mkdir(exist_ok=True, parents=True)
    logging_dir = Path('tensorboard_training') / Path(args.output_dir)
    logging_dir.mkdir(parents=True, exist_ok=True)

    training_args = Seq2SeqTrainingArguments(
        predict_with_generate=True,
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epoch,
        save_steps=50,
        logging_strategy=IntervalStrategy.STEPS,
        evaluation_strategy=IntervalStrategy.STEPS,
        logging_steps=5,
        eval_steps=5,
        logging_dir=str(logging_dir),
        save_strategy=IntervalStrategy.STEPS,
        save_total_limit=1,
        do_eval=True,
        eval_accumulation_steps=2,
        eval_delay=0.0,

    )

    model, tokenizer = prep_model(args.model_name)
    config = DATASETS_CONFIGS[args.dataset_config]
    train_dataset, validation_dataset, _ = prep_dataset(
        config["dataset_name"],
        config["dataset_config"],
        tokenizer,
        train_max_size=100,
        validation_max_size=50,
        test_max_size=100,
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

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
    )

    trainer.evaluate()
    trainer.train()
