import argparse
import json
from pathlib import Path
from typing import List

import torch
from tqdm import tqdm

from Todd import ScorerType, MahalanobisScorer
from toddbenchmark.classification_datasets import prep_model
from toddbenchmark.classification_datasets_configs import (
    DATASETS_CONFIGS,
    load_requested_dataset,
)
from toddbenchmark.utils_classification import (
    prepare_detectors,
    evaluate_dataloader,
)

from toddbenchmark.utils import mk_file_name


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a model on a dataset")
    parser.add_argument("--model_name", type=str, default="distilbert-base-uncased")

    config_choices: List[str] = list(DATASETS_CONFIGS.keys())

    parser.add_argument(
        "--in_config",
        type=str,
        default="sst2",
        choices=config_choices,
    )
    parser.add_argument(
        "--out_configs",
        type=str,
        nargs="+",
        default=["imdb"],
        choices=config_choices,
    )

    parser.add_argument("--batch_size", type=int, default=128)

    parser.add_argument("--seed", type=int, default=42)

    # Dataset max sizes
    parser.add_argument(
        "--validation_size",
        type=int,
        default=30,
        help="Max size of validation set used as reference to fit detectors",
    )
    parser.add_argument(
        "--test_size",
        type=int,
        default=30,
        help="Max size of test set to evaluate detectors",
    )

    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--output_dir", type=str, default="output")
    return parser.parse_args()

# Instantiate the detectors
detectors: List[ScorerType] = [MahalanobisScorer(layers=[-1])]


if __name__ == "__main__":
    args = parse_args()

    # Load model and tokenizer
    model, tokenizer = prep_model(args.model_name)

    # Load the reference set
    _, validation_loader, test_loader = load_requested_dataset(
        args.in_config, tokenizer, 0, args.validation_size, args.test_size
    )

    # Fit the detectors on the behavior of the model on the (in) validation set
    detectors = prepare_detectors(detectors, model, validation_loader, tokenizer)

    # ====================== Evaluate the detectors on the (in) validation set ====================== #

    # Evaluate the model on the (in) validation set:
    print("Evaluating on the in-distribution validation set")
    records = evaluate_dataloader(
        model,
        validation_loader,
        tokenizer,
        detectors,
    )

    inval_ds_scores_path = Path(args.output_dir) / (
        "validation_scores/"
        + mk_file_name(args.model_name, args.in_config, args.in_config)
    )
    inval_ds_scores_path.parent.mkdir(parents=True, exist_ok=True)

    with open(inval_ds_scores_path, "w") as f:
        json.dump(records, f)

    # ====================== Evaluate the detectors on the (in) test set ====================== #

    # Evaluate the model on the (in) test set
    print("Evaluating on the in-distribution test set")
    records = evaluate_dataloader(
        model,
        test_loader,
        tokenizer,
        detectors,
    )

    in_ds_scores_path = Path(args.output_dir) / (
        "test_scores/" + mk_file_name(args.model_name, args.in_config, args.in_config)
    )
    in_ds_scores_path.parent.mkdir(parents=True, exist_ok=True)

    with open(in_ds_scores_path, "w") as f:
        json.dump(records, f)

    # ====================== Evaluate the detectors on the (out) test sets ====================== #

    print("BEGIN OOD EVALUATION")
    for out_config in tqdm(args.out_configs):
        # Load the out-of-distribution set
        _, _, test_loader = load_requested_dataset(
            out_config, tokenizer, 0, 0, args.test_size
        )

        # Evaluate the model on the (out) test set
        print("Evaluating on the out-of-distribution test set")
        records = evaluate_dataloader(
            model,
            test_loader,
            tokenizer,
            detectors,
        )

        out_ds_scores_path = Path(args.output_dir) / (
            "test_scores/" + mk_file_name(args.model_name, args.in_config, out_config)
        )
        out_ds_scores_path.parent.mkdir(parents=True, exist_ok=True)

        with open(out_ds_scores_path, "w") as f:
            json.dump(records, f)
