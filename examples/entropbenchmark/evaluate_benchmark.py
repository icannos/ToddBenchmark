import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import torch
from tqdm import tqdm
from transformers import GenerationConfig

from Todd import (
    ScorerType,
    MahalanobisScorer,
    SequenceRenyiNegScorer,
    SequenceFisherRaoScorer,
    BeamRenyiInformationProjection,
    CosineProjectionScorer,
    InformationProjection,
    DataDepthScorer,
    BestBeamSeqRenyi,
    BestBeamSeqFisherRao,
    BestBeamMSPScorer,
    BestBeamSoftMaxEnergyScorer,
    BestBeamInformationProjection,
    SoftMaxEnergyScorer,
    SequenceMSPScorer,
)
from toddbenchmark.generation_datasets import prep_model
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

from toddbenchmark.utils import dump_pickle

from generation_args import GENERATION_CONFIGS

from rouge_score import rouge_scorer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate generation with a reference distribution on different datasets"
    )
    parser.add_argument("--model_name", type=str, default="Helsinki-NLP/opus-mt-de-en")

    config_choices: List[str] = list(DATASETS_CONFIGS.keys())

    parser.add_argument(
        "--in_config",
        type=str,
        default="tatoeba_mt_deu_eng",
        choices=config_choices,
    )
    parser.add_argument(
        "--out_configs",
        type=str,
        nargs="+",
        default=[],
        choices=config_choices,
    )

    parser.add_argument("--batch_size", type=int, default=2)
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
    parser.add_argument(
        "--output_dir", type=str, default="results/instruction_qa_sampling"
    )

    parser.add_argument(
        "--generation_config",
        type=str,
        default="sampling",
        help="Name of the generation config to use.",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        default=False,
        help="Append to existing results",
    )

    parser.add_argument(
        "--instruction",
        type=str,
        default="",
        help="Instruction to use for generation.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    bleuscorer = BLEU(effective_order=True)
    bertscorer = BERTScorer(lang="en", rescale_with_baseline=True)

    rougescorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL", "rougeLsum"],
        split_summaries=".",
        use_stemmer=True,
    )

    def metric_eval(prediction, reference):
        bleu = bleuscorer.sentence_score(hypothesis=prediction, references=[reference])
        bert = bertscorer.score([prediction], [reference])
        rouge = rougescorer.score(prediction, reference)

        rouge_scores = {}

        for rouge, score in rouge.items():
            rouge_scores[f"{rouge}_precision"] = score.precision
            rouge_scores[f"{rouge}_recall"] = score.recall
            rouge_scores[f"{rouge}_fmeasure"] = score.fmeasure

        return {
            "bleu": bleu.score,
            "bert": bert[2][0].cpu().detach().tolist(),
            **rouge_scores,
        }

    METADATA = {
        "model_name": args.model_name,
        "in_config": args.in_config,
        "generation_config_name": args.generation_config,
        "generation_config": GENERATION_CONFIGS[args.generation_config],
        "instruction": args.instruction,
        "seed": args.seed,
        "output_dir": args.output_dir,
    }

    # Load model and tokenizer
    model, tokenizer = prep_model(args.model_name)
    model.to(args.device)

    gen_config = GenerationConfig(**GENERATION_CONFIGS[args.generation_config])

TEMPERATURES = [
    0.1,
    0.25,
    0.5,
    1,
    1.25,
    1.5,
    2,
    2.25,
    2.5,
    3,
    3.25,
    3.5,
    4,
    4.25,
    4.5,
    5,
]

ALPHAS = [round(a, 3) if a != 0 else 0.05 for a in np.arange(0, 2, 0.1) if a != 1] + [5]

detectors: List[ScorerType] = [
    SequenceFisherRaoScorer(
        alpha=a,
        temperature=t,
        mode="token",  # input, output, token
        num_return_sequences=GENERATION_CONFIGS[args.generation_config][
            "num_return_sequences"
        ],
        num_beam=GENERATION_CONFIGS[args.generation_config]["num_beams"],
    )
    for t in TEMPERATURES
    for a in ALPHAS
]

detectors += [
    SequenceRenyiNegScorer(
        alpha=a,
        temperature=t,
        mode="token",  # input, output, token
        num_return_sequences=GENERATION_CONFIGS[args.generation_config][
            "num_return_sequences"
        ],
        num_beam=GENERATION_CONFIGS[args.generation_config]["num_beams"],
    )
    for t in TEMPERATURES
    for a in ALPHAS
]

detectors += [
    BeamRenyiInformationProjection(
        alpha=a,
        temperature=t,
        use_soft_projection=True,
        n_neighbors=n,
        pad_token_id=tokenizer.pad_token_id,
        num_return_sequences=GENERATION_CONFIGS[args.generation_config][
            "num_return_sequences"
        ],
        num_beams=GENERATION_CONFIGS[args.generation_config]["num_beams"],
        mode="output",
    )
    for t in TEMPERATURES
    for a in ALPHAS
    # step of 2, until num_return_sequences
    for n in range(
        2, GENERATION_CONFIGS[args.generation_config]["num_return_sequences"], 2
    )
]

detectors += [
    # InformationProjection(
    #     alpha=0.5,
    #     temperature=1,
    #     use_soft_projection=True,
    #     n_neighbors=8,
    #     pad_token_id=tokenizer.pad_token_id,
    #     num_return_sequences=GENERATION_CONFIGS[args.generation_config][
    #         "num_return_sequences"
    #     ],
    #     num_beams=GENERATION_CONFIGS[args.generation_config]["num_beams"],
    #     mode="output",
    # ),
    SoftMaxEnergyScorer(
        mode="input",
        num_beams=GENERATION_CONFIGS[args.generation_config]["num_beams"],
        num_return_sequences=GENERATION_CONFIGS[args.generation_config][
            "num_return_sequences"
        ],
    ),
    SequenceMSPScorer(
        mode="input",
        num_beams=GENERATION_CONFIGS[args.generation_config]["num_beams"],
        num_return_sequences=GENERATION_CONFIGS[args.generation_config][
            "num_return_sequences"
        ],
    ),
]

detectors.extend([MahalanobisScorer(), CosineProjectionScorer(), DataDepthScorer()])


def add_instruction_token(sample):
    sample["source"] = f"{args.instruction}{sample['source']}"
    return sample


# Load the reference set

_, validation_loader, test_loader = load_requested_dataset(
    args.in_config,
    tokenizer,
    args.batch_size,
    0,
    args.validation_size,
    args.test_size,
    update_input_fn=add_instruction_token,
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
    metric_eval=metric_eval,
    generation_config=gen_config,
)

inval_ds_scores_path = Path(args.output_dir) / (
    "validation_scores/" + mk_file_name(args.model_name, args.in_config, args.in_config)
)
inval_ds_scores_path.parent.mkdir(parents=True, exist_ok=True)

records["metadata"] = METADATA
records["metadata"]["dataset_in"] = args.in_config
records["metadata"]["dataset_out"] = args.in_config

dump_pickle(records, inval_ds_scores_path)

# ====================== Evaluate the detectors on the (in) test set ====================== #

# Evaluate the model on the (in) test set
print("Evaluating on the in-distribution test set")

records = evaluate_dataloader(
    model,
    test_loader,
    tokenizer,
    detectors,
    generation_config=gen_config,
    metric_eval=metric_eval,
)

in_ds_scores_path = Path(args.output_dir) / (
    "test_scores/" + mk_file_name(args.model_name, args.in_config, args.in_config)
)
in_ds_scores_path.parent.mkdir(parents=True, exist_ok=True)

records["metadata"] = METADATA
records["metadata"]["dataset_in"] = args.in_config
records["metadata"]["dataset_out"] = args.in_config

dump_pickle(records, in_ds_scores_path)

# ====================== Evaluate the detectors on the (out) test sets ====================== #
if args.out_configs is None:
    # exit:
    exit(0)

print("BEGIN OOD EVALUATION")

for out_config in tqdm(args.out_configs):
    # Load the out-of-distribution set
    _, _, test_loader = load_requested_dataset(
        out_config,
        tokenizer,
        args.batch_size,
        0,
        0,
        args.test_size,
        update_input_fn=add_instruction_token,
    )

    # Evaluate the model on the (out) test set
    print("Evaluating on the out-of-distribution test set")
    records = evaluate_dataloader(
        model,
        test_loader,
        tokenizer,
        detectors,
        generation_config=gen_config,
        metric_eval=metric_eval,
    )

    out_ds_scores_path = Path(args.output_dir) / (
        "test_scores/" + mk_file_name(args.model_name, args.in_config, out_config)
    )
    out_ds_scores_path.parent.mkdir(parents=True, exist_ok=True)

    records["metadata"] = METADATA
    records["metadata"]["dataset_in"] = args.in_config
    records["metadata"]["dataset_out"] = out_config

    dump_pickle(records, out_ds_scores_path)
