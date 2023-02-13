import argparse
import json
from pathlib import Path
from typing import List
from rouge_score import rouge_scorer

import torch
from tqdm import tqdm

from Todd import (
    ScorerType,
    MahalanobisScorer,
    SequenceRenyiNegScorer,
    BeamRenyiInformationProjection,
    CosineProjectionScorer, SequenceRenyiNegDataFittedScorer,
)
from toddbenchmark.generation_datasets import prep_model
from toddbenchmark.generation_datasets_configs import (
    DATASETS_CONFIGS,
    load_requested_dataset,
)
from toddbenchmark.utils_generation import (
    prepare_detectors,
    evaluate_dataloader,
    mk_file_name, prepare_idf,
)
import evaluate
from sacrebleu import BLEU
from bert_score import BERTScorer

from toddbenchmark.utils import dump_json


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a model on a dataset")
    parser.add_argument("--model_name", type=str, default="Helsinki-NLP/opus-mt-de-en")

    config_choices: List[str] = list(DATASETS_CONFIGS.keys())

    parser.add_argument(
        "--in_config",
        type=str,
        default="tatoeba_mt_deu_eng",
        choices=config_choices,
    )

    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_return_sequences", type=int, default=2)

    parser.add_argument("--max_length", type=int, default=150)
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
    parser.add_argument("--output_dir", type=str, default="results/summarization")
    parser.add_argument(
        "--append",
        action="store_true",
        default=False,
        help="Append to existing results",
    )
    return parser.parse_args()


# detectors: List[FilterType] = [MahalanobisFilter(threshold=0.5, layers=[-1])]


if __name__ == "__main__":
    args = parse_args()

    bleuscorer = BLEU(effective_order=True)
    bertscorer = BERTScorer(lang="en", rescale_with_baseline=True)
    rougescorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL', 'rougeLsum'], split_summaries='.',
                                           use_stemmer=True)

    def metric_eval(prediction, reference):
        bleu = bleuscorer.sentence_score(hypothesis=prediction, references=[reference])
        bert = bertscorer.score([prediction], [reference])
        rouge = rougescorer.score(prediction, reference)

        rouge_scores = {}

        for rouge, score in rouge.items():
            rouge_scores[f"{rouge}_precision"] = score.precision
            rouge_scores[f"{rouge}_recall"] = score.recall
            rouge_scores[f"{rouge}_fmeasure"] = score.fmeasure

        return {"bleu": bleu.score, "bert": bert[2][0].cpu().detach().tolist(), **rouge_scores}


    # Load model and tokenizer
    model, tokenizer = prep_model(args.model_name)

    # Load the reference set

    _, validation_loader, test_loader = load_requested_dataset(
        args.in_config,
        tokenizer,
        args.batch_size,
        0,
        args.validation_size,
        args.test_size,
    )

    # idf = prepare_idf(tokenizer, model, validation_loader)

    detectors: List[ScorerType] = [
        SequenceRenyiNegScorer(
            alpha=a,
            temperature=t,
            mode="token",  # input, output, token
            num_return_sequences=args.num_return_sequences,
            num_beam=args.num_return_sequences,
        )
        for t in [0.1, 0.5, 1, 1.5, 2, 3, 4, 5]
        for a in [0.1, 0.5, 0.9, 1.1, 1.5, 2, 3, 4.0, 5.0]
    ]

    detectors.extend([MahalanobisScorer(), CosineProjectionScorer()])

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
        max_length=150,
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
        max_length=50,
        metric_eval=metric_eval,
    )

    records["out_config"] = args.in_config
    records["in_config"] = args.in_config

    in_ds_scores_path = Path(args.output_dir) / (
            "test_scores/" + mk_file_name(args.model_name, args.in_config, args.in_config)
    )
    in_ds_scores_path.parent.mkdir(parents=True, exist_ok=True)

    dump_json(records, in_ds_scores_path, append=args.append)
