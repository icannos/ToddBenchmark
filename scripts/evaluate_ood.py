import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import torch
from datasets import load_metric

from Todd import FilterType
from toddbenchmark.datasets_configs import DATASETS_CONFIGS, load_requested_dataset
from toddbenchmark.generation_data import prep_model
from toddbenchmark.utils import prepare_detectors, evaluate_dataloader, mk_file_name


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a model on a dataset")
    parser.add_argument("--model_name", type=str, default="Helsinki-NLP/opus-mt-en-de")

    config_choices: List[str] = list(DATASETS_CONFIGS.keys())

    parser.add_argument(
        "--in_config",
        type=str,
        default="wmt16_de_en",
        choices=config_choices,
    )
    parser.add_argument(
        "--out_configs",
        type=str,
        nargs="+",
        default="tatoeba_mt_deu_eng",
        choices=config_choices,
    )

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_return_sequences", type=int, default=1)

    parser.add_argument("--max_length", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--output_dir", type=str, default="output")
    return parser.parse_args()


detectors: List[FilterType] = []


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

    model, tokenizer = prep_model(args.model_name)

    # Load the reference set
    _, validation_loader, test_loader = load_requested_dataset(
        args.in_config, tokenizer
    )

    # Fit the detectors on the behavior of the model on the (in) validation set
    detectors = prepare_detectors(detectors, model, validation_loader)

    # Evaluate the model on the (in) validation set:
    print("Evaluating on the in-distribution validation set")
    records = evaluate_dataloader(
        model,
        validation_loader,
        tokenizer,
        detectors,
        num_beams=4,
        num_return_sequences=4,
        max_length=150,
    )

    inval_ds_scores_path = Path(args.output_dir) / (
        "validation_scores/"
        + mk_file_name(args.model_name, args.in_config, args.in_config)
    )

    with open(inval_ds_scores_path, "w") as f:
        json.dump(records, f)

    # Evaluate the model on the (in) test set
    print("Evaluating on the in-distribution test set")
    records = evaluate_dataloader(
        model,
        test_loader,
        tokenizer,
        detectors,
        num_beams=4,
        num_return_sequences=4,
        max_length=150,
    )

    in_ds_scores_path = Path(args.output_dir) / (
        "test_scores/" + mk_file_name(args.model_name, args.in_config, args.in_config)
    )

    with open(in_ds_scores_path, "w") as f:
        json.dump(records, f)

    print("BEGIN OOD EVALUATION")
    for out_config in args.out_configs:
        # Load the out-of-distribution set
        _, _, test_loader = load_requested_dataset(out_config, tokenizer)

        # Evaluate the model on the (out) test set
        print("Evaluating on the out-of-distribution test set")
        records = evaluate_dataloader(
            model,
            test_loader,
            tokenizer,
            detectors,
            num_beams=4,
            num_return_sequences=4,
            max_length=150,
        )

        out_ds_scores_path = Path(args.output_dir) / (
            "test_scores/" + mk_file_name(args.model_name, args.in_config, out_config)
        )

        with open(out_ds_scores_path, "w") as f:
            json.dump(records, f)
