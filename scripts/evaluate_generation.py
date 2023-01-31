import argparse
import json
from pathlib import Path
from typing import List

import torch
from tqdm import tqdm
import configue

from Todd import (
    ScorerType,
    MahalanobisScorer,
    CosineProjectionScorer,
    SequenceRenyiNegScorer,
    SequenceRenyiNegDataFittedScorer,
    BeamRenyiInformationProjection,
)

from toddbenchmark.generation_datasets_configs import (
    DATASETS_CONFIGS,
    load_requested_dataset,
)
from toddbenchmark.utils_generation import (
    prepare_detectors,
    evaluate_dataloader,
    mk_file_name,
)
import evaluate
from sacrebleu import BLEU
from bert_score import BERTScorer

from toddbenchmark.utils import dump_json


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a model on a dataset")
    parser.add_argument("--model_name", type=str, default="Helsinki-NLP/opus-mt-de-en")


    parser.add_argument(
        "--in_config",
        type=str,
    )
    parser.add_argument(
        "--out_configs",
        type=str,
        nargs="+",
    )

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_return_sequences", type=int, default=1)

    parser.add_argument("--max_length", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)

    # Dataset max sizes
    parser.add_argument(
        "--validation_size",
        type=int,
        default=100,
        help="Max size of validation set used as reference to fit detectors",
    )
    parser.add_argument(
        "--test_size",
        type=int,
        default=300,
        help="Max size of test set to evaluate detectors",
    )

    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument(
        "--append",
        action="store_true",
        default=False,
        help="Append to existing results",
    )
    parser.add_argument(
        "--model_config_path",
        type=str,
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config: dict = configue.load(args.model_config_path)

    # Load model
    model = config["model"]
    tokenizer = config["tokenizer"]

    bertscorer: BERTScorer = config["bert_scorer"]
    bleuscorer: BLEU = config["bleu_scorer"]

    def metric_eval(prediction, reference):
        bleu = bleuscorer.sentence_score(hypothesis=prediction, references=[reference])
        bert = bertscorer.score([prediction], [reference])
        return {"bleu": bleu.score, "bert": bert[2][0].cpu().detach().tolist()}

    detectors:  List[ScorerType] = config["detectors"]
    detectors.append(SequenceRenyiNegDataFittedScorer(
            alpha=2,
            temperature=1,
            mode="input",  # mode="token",  # input, output, token
            num_return_sequences=args.num_return_sequences,
            num_beam=args.num_return_sequences,
        ))

    detectors.extend([
        SequenceRenyiNegScorer(
            alpha=a,
            temperature=t,
            mode="input",  # mode="token",  # input, output, token
            num_return_sequences=args.num_return_sequences,
            num_beam=args.num_return_sequences,
        )
        for t in [0.5, 1, 1.5, 2, 5]
        for a in [0.05, 0.1, 0.5, 0.9, 1.1, 1.5, 2, 3]
    ])

    detectors.extend([BeamRenyiInformationProjection(
        alpha=a,
        num_return_sequences=args.num_return_sequences,
        num_beams=args.num_return_sequences,
        mode="output",
    ) for a in [0.05, 0.1, 0.5, 0.9, 1.1, 1.5, 2, 3]])

    # Load the reference set
    _, validation_loader, test_loader = load_requested_dataset(
        args.in_config,
        tokenizer,
        args.batch_size,
        0,
        args.validation_size,
        args.test_size,
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
        num_beams=args.num_return_sequences,
        num_return_sequences=args.num_return_sequences,
        max_length=200,
        metric_eval=metric_eval,
    )

    inval_ds_scores_path = Path(args.output_dir) / (
        "validation_scores/"
        + mk_file_name(args.model_name, args.in_config, args.in_config)
    )
    inval_ds_scores_path.parent.mkdir(parents=True, exist_ok=True)

    dump_json(records, inval_ds_scores_path, append=args.append)

    # ====================== Evaluate the detectors on the (in) test set ====================== #

    # Evaluate the model on the (in) test set
    print("Evaluating on the in-distribution test set")
    records = evaluate_dataloader(
        model,
        test_loader,
        tokenizer,
        detectors,
        num_beams=args.num_return_sequences,
        num_return_sequences=args.num_return_sequences,
        max_length=150,
        metric_eval=metric_eval,
    )

    in_ds_scores_path = Path(args.output_dir) / (
        "test_scores/" + mk_file_name(args.model_name, args.in_config, args.in_config)
    )
    in_ds_scores_path.parent.mkdir(parents=True, exist_ok=True)

    dump_json(records, in_ds_scores_path, append=args.append)

    # ====================== Evaluate the detectors on the (out) test sets ====================== #

    print("BEGIN OOD EVALUATION")
    for out_config in tqdm(args.out_configs):
        # Load the out-of-distribution set
        _, _, test_loader = load_requested_dataset(
            out_config, tokenizer, args.batch_size, 0, 0, args.test_size
        )

        # Evaluate the model on the (out) test set
        print("Evaluating on the out-of-distribution test set")
        records = evaluate_dataloader(
            model,
            test_loader,
            tokenizer,
            detectors,
            num_beams=args.num_return_sequences,
            num_return_sequences=args.num_return_sequences,
            max_length=150,
            metric_eval=metric_eval,
        )

        out_ds_scores_path = Path(args.output_dir) / (
            "test_scores/" + mk_file_name(args.model_name, args.in_config, out_config)
        )
        out_ds_scores_path.parent.mkdir(parents=True, exist_ok=True)

        dump_json(records, out_ds_scores_path, append=args.append)
