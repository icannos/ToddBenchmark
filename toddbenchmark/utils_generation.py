from collections import defaultdict, Counter
from math import log
from typing import List, Dict, Optional, Callable

import torch
from datasets import load_dataset, DatasetDict
from torch.utils.data import DataLoader
from transformers import GenerationConfig
from tqdm import tqdm

from Todd import ScorerType


def prepare_detectors(
    detectors: List[ScorerType],
    model,
    loader: DataLoader,
    tokenizer,
    force_bos_token_id=None,
) -> List[ScorerType]:
    """
    Fit the detectors on the behavior of the model on the (in) validation set
    :param detectors: List of detectors to fit
    :param model: Model to evaluate
    :param loader: Dataloader (reference set) to evaluate the model on
    :return: List of fitted detectors
    """

    for batch in loader:

        inputs = tokenizer(
            batch["source"], padding=True, truncation=True, return_tensors="pt"
        )
        labels = tokenizer(
            batch["target"], padding=True, truncation=True, return_tensors="pt"
        )

        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        output = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=100,
            num_beams=2,
            num_return_sequences=2,
            early_stopping=True,
            return_dict_in_generate=True,
            output_scores=True,
            output_hidden_states=True,
            force_bos_token_id=force_bos_token_id,
        )

        for detector in detectors:
            detector.accumulate(output)

    for detector in detectors:
        detector.fit()

    return detectors


def flatten_dict(d):
    """
    Flatten a dictionary
    :param d: Dictionary to flatten
    :return: Flattened dictionary
    """
    result = {}
    for key, value in d.items():
        if isinstance(value, dict):
            result.update({f"{key}+{k}": v for k, v in flatten_dict(value)})
        else:
            result[key] = value

    return result


def evaluate_batch(output, detectors: List[ScorerType]) -> Dict[str, torch.Tensor]:

    scores = {}
    for detector in detectors:
        s = detector.compute_scores_benchmark(output)
        scores |= {f"{detector}+{k}": v for k, v in flatten_dict(s).items()}

    return scores


def prepare_idf(
    tokenizer,
    model,
    loader: DataLoader,
):

    input_refs = []
    for batch in loader:
        inputs = tokenizer(
            batch["source"], padding=True, truncation=True, return_tensors="pt"
        )
        input_refs.extend(inputs.input_ids.tolist())

    # Mock generation to get the vocab size
    inputs = inputs.to(model.device)
    vocab_size = (
        model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=16,
            return_dict_in_generate=True,
            output_scores=True,
        )
        .scores[0]
        .shape[1]
    )
    idf_count = Counter()
    num_docs = len(input_refs)

    idf_count.update(sum([list(set(i)) for i in input_refs], []))

    idf_dict = defaultdict(lambda: log((num_docs + 1) / (1)))
    idf_dict.update(
        {idx: log((num_docs + 1) / (c + 1)) for (idx, c) in idf_count.items()}
    )

    idf = torch.ones(vocab_size, dtype=torch.float) * (log((num_docs + 1) / (0 + 1)))
    for idx, c in idf_dict.items():
        idf[idx] = c
    return idf / idf.sum()


def evaluate_dataloader(
    model,
    data_loader: DataLoader,
    tokenizer,
    detectors: List[ScorerType],
    generation_config: GenerationConfig,
    metric_eval: Optional[Callable] = None,
) -> Dict[str, List]:

    num_return_sequences = generation_config.num_return_sequences

    # Initialize the scores dictionary
    records: Dict[str, List] = {
        f"{detector}+{score_name}": []
        for detector in detectors
        for score_name in detector.score_names
    }

    records["likelihood"] = []
    with torch.no_grad():
        for batch_idx, batch in tqdm(enumerate(data_loader)):
            inputs = tokenizer(
                batch["source"], padding=True, truncation=True, return_tensors="pt"
            )
            labels = tokenizer(
                batch["target"], padding=True, truncation=True, return_tensors="pt"
            )

            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            output = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                generation_config=generation_config,
            )

            # Should be a dictionary with keys ood scores,
            # each containing a numpy array of shape (batch_size, num_return_sequences))

            ood_scores = evaluate_batch(output, detectors)
            ood_scores = {
                k: (scores.tolist() if not isinstance(scores, list) else scores)
                for k, scores in ood_scores.items()
            }

            for k, scores in ood_scores.items():
                records[k].extend(scores)

            # A list of list ie each returned sequence for each batch
            generated_sequences = output.sequences.view(
                output.sequences.shape[0] // num_return_sequences,
                num_return_sequences,
                -1,
            )

            global_perfs_scores = defaultdict(list)
            for sample_id, seqs in enumerate(generated_sequences):
                decoded_sequences = tokenizer.batch_decode(
                    seqs,
                    skip_special_tokens=True,
                )

                per_gen_score = defaultdict(list)
                for hyp in decoded_sequences:
                    for k, v in metric_eval(hyp, batch["target"][sample_id]).items():
                        per_gen_score[k].append(v)
                    per_gen_score["hyp"].append(hyp)

                for k, v in per_gen_score.items():
                    global_perfs_scores[k].append(v)
                global_perfs_scores["ref"].append(batch["target"][sample_id])
                global_perfs_scores["source"].append(batch["source"][sample_id])

            for k, v in global_perfs_scores.items():
                if k not in records:
                    records[k] = []
                records[k].extend(v)

            if hasattr(output, "sequences_scores"):
                sequences_scores = output.sequences_scores.view(
                    output.sequences_scores.shape[0] // num_return_sequences,
                    num_return_sequences,
                ).tolist()
            else:
                sequences_scores = model.compute_transition_scores(
                    output.sequences,
                    output.scores,
                    beam_indices=(
                        output.beam_indices if hasattr(output, "beam_indices") else None
                    ),
                    normalize_logits=False,
                )

                sequences_scores = (
                    sequences_scores.sum(-1)
                    .view(
                        sequences_scores.shape[0] // num_return_sequences,
                        num_return_sequences,
                    )
                    .tolist()
                )

            records["likelihood"].extend(sequences_scores)

    return records


def try_load_dataset_config(dataset_name: str, dataset_config: str) -> DatasetDict:
    src, tgt = dataset_config.split("-")

    # Try to load the dataset from the datasets library with one config or its permutation
    try:
        dataset = load_dataset(
            dataset_name,
            dataset_config,
            # download_mode="force_redownload",
            verification_mode="no_checks",
        )
    except Exception:
        dataset_config = tgt + "-" + src
        src, tgt = dataset_config.split("-")

        try:
            dataset = load_dataset(
                dataset_name,
                dataset_config,
                # download_mode="force_redownload",
                verification_mode="no_checks",
            )
        except Exception:
            raise ValueError(
                "Invalid dataset config. None of the following configs are valid: "
                + dataset_config
                + ", "
                + tgt
                + "-"
                + src
            )

    return dataset


def sanitize_model_name(model_name: str) -> str:
    """
    Sanitize the model name to be used as a file name
    :param model_name: Model name
    :return: Sanitized model name
    """
    return model_name.replace("/", "_")


def mk_file_name(model_name: str, dataset_in_config, dataset_out_config) -> str:
    """
    Make a file name for the results
    :param model_name: Model name
    :param dataset_in_config: Dataset in config
    :param dataset_out_config: Dataset out config
    :return: File name
    """
    model_name = sanitize_model_name(model_name)
    return f"{model_name}_{dataset_in_config}_{dataset_out_config}.json"


def task_performance_evaluation():
    pass
