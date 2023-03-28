import argparse
import os
from pathlib import Path
from typing import List

from tqdm import tqdm
import torch
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
    prepare_detectors_out,
    fit_models,
    evaluate_dataloader,
)
from sacrebleu import BLEU
from bert_score import BERTScorer

from toddbenchmark.utils import dump_json


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a model on a dataset")
    parser.add_argument(
        "--in_config",
        type=str,
    )
    parser.add_argument(
        "--out_configs",
        type=str,
        nargs="+",
    )

    parser.add_argument(
        "--experiment_config_path",
        type=str,
    )
    return parser.parse_args()


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    args = parse_args()
    config: dict = configue.load(args.experiment_config_path)

    # Load model
    experiment_args = config["experiment_args"]
    model = config["model"]
    if not model.kwargs.get("load_in_8_bit", None):
        model.to(experiment_args.device)
    model.eval()

    tokenizer = config["tokenizer"]

    bertscorer: BERTScorer = config["bert_scorer"]
    bleuscorer: BLEU = config["bleu_scorer"]

    def metric_eval(prediction, reference):
        bleu = bleuscorer.sentence_score(hypothesis=prediction, references=[reference])
        bert = bertscorer.score([prediction], [reference])
        return {"bleu": bleu.score, "bert": bert[2][0].item()}

    # Load the reference set
    _, validation_loader, test_loader = load_requested_dataset(
        args.in_config,
        tokenizer,
        experiment_args.batch_size,
        0,
        experiment_args.validation_size,
        experiment_args.test_size,
    )
    ref_probs = fit_models(tokenizer, model, validation_loader)

    detectors:  List[ScorerType] = config["detectors"]
    detectors.extend([SequenceRenyiNegDataFittedScorer(
        alpha=a,
        temperature=t,
        mode="input",  # mode="token",  # input, output, token
        num_return_sequences=experiment_args.num_return_sequences,
        num_beam=experiment_args.num_return_sequences,
        reference_vocab_distribution=ref_probs.to(model.device),
    )
        for t in [0.5, 1, 2, 5]
        for a in [0.05, 0.1, 0.5, 2, 3]
    ])

    detectors.extend([
        SequenceRenyiNegScorer(
            alpha=a,
            temperature=t,
            mode="input",  # mode="token",  # input, output, token
            num_return_sequences=experiment_args.num_return_sequences,
            num_beam=experiment_args.num_return_sequences,
        )
        for t in [0.5, 1,  2, 5]
        for a in [0.05, 0.1, 0.5, 1, 2, 3]
    ])

    # detectors.extend([BeamRenyiInformationProjection(
    #     alpha=a,
    #     num_return_sequences=experiment_args.num_return_sequences,
    #     num_beams=experiment_args.num_return_sequences,
    #     mode="output",
    # ) for a in [0.05, 0.1, 0.5, 0.9, 1.1, 1.5, 2, 3]])

    # Fit the detectors on the behavior of the model on the (in) validation set
    detectors = prepare_detectors(detectors, model, validation_loader, tokenizer)

    # For the classifier scorers, we need to fit them on the (out) validation set
    _, validation_loader_out, _ = load_requested_dataset(
        args.out_configs[0],
        tokenizer,
        experiment_args.batch_size,
        0,
        experiment_args.validation_size,
        experiment_args.test_size,
    )
    detectors = prepare_detectors_out(detectors, model, validation_loader_out, tokenizer)
    del validation_loader_out



    # ====================== Evaluate the detectors on the (in) validation set ====================== #

    # Evaluate the model on the (in) validation set:
    print("Evaluating on the in-distribution validation set")
    records = evaluate_dataloader(
        model,
        validation_loader,
        tokenizer,
        detectors,
        num_beams=experiment_args.num_return_sequences,
        num_return_sequences=experiment_args.num_return_sequences,
        max_length=experiment_args.max_length,
        metric_eval=metric_eval,
    )

    reference_file_name = args.in_config.split("/")[-1].split(".")[0]
    dump_path = Path(os.path.join(experiment_args.output_dir, "validation_scores", f"{reference_file_name}.json"))
    dump_path.parent.mkdir(parents=True, exist_ok=True)
    dump_json(records, dump_path, append=experiment_args.append)

    # ====================== Evaluate the detectors on the (in) test set ====================== #

    # Evaluate the model on the (in) test set
    print("Evaluating on the in-distribution test set")
    records = evaluate_dataloader(
        model,
        test_loader,
        tokenizer,
        detectors,
        num_beams=experiment_args.num_return_sequences,
        num_return_sequences=experiment_args.num_return_sequences,
        max_length=experiment_args.max_length,
        metric_eval=metric_eval,
    )

    reference_file_name = args.in_config.split("/")[-1].split(".")[0]
    dump_path = Path(os.path.join(experiment_args.output_dir, "test_scores", f"{reference_file_name}.json"))
    dump_path.parent.mkdir(parents=True, exist_ok=True)
    dump_json(records, dump_path, append=experiment_args.append)

    # ====================== Evaluate the detectors on the (out) test sets ====================== #

    print("BEGIN OOD EVALUATION")
    for out_config in tqdm(args.out_configs):
        # Load the out-of-distribution set
        _, _, test_loader = load_requested_dataset(
            out_config, tokenizer, experiment_args.batch_size, 0, 0, experiment_args.test_size
        )

        # Evaluate the model on the (out) test set
        print("Evaluating on the out-of-distribution test set")
        records = evaluate_dataloader(
            model,
            test_loader,
            tokenizer,
            detectors,
            num_beams=experiment_args.num_return_sequences,
            num_return_sequences=experiment_args.num_return_sequences,
            max_length=experiment_args.max_length,
            metric_eval=metric_eval,
        )

        reference_file_name = out_config.split("/")[-1].split(".")[0]
        dump_path = Path(os.path.join(experiment_args.output_dir, "test_scores", f"{reference_file_name}.json"))
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        dump_json(records, dump_path, append=experiment_args.append)
