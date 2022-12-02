from typing import List, Dict

import torch
from datasets import load_dataset, DatasetDict
from torch.utils.data import DataLoader

from Todd import FilterType


def prepare_detectors(
    detectors: List[FilterType], model, loader: DataLoader
) -> List[FilterType]:
    """
    Fit the detectors on the behavior of the model on the (in) validation set
    :param detectors: List of detectors to fit
    :param model: Model to evaluate
    :param loader: Dataloader (reference set) to evaluate the model on
    :return: List of fitted detectors
    """

    for batch in loader:
        output = model.generate(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            max_length=100,
            num_beams=4,
            num_return_sequences=4,
            early_stopping=True,
            return_dict_in_generate=True,
            output_scores=True,
            output_hidden_states=True,
        )

        for detector in detectors:
            detector.accumulate(output)

    for detector in detectors:
        detector.fit()

    return detectors


def evaluate_batch(output, detectors: List[FilterType]) -> Dict[str, torch.Tensor]:

    scores = {}
    for detector in detectors:
        scores[f"{detector}"] = detector.compute_scores_benchmark(output)

    return scores


def evaluate_dataloader(
    model,
    data_loader: DataLoader,
    tokenizer,
    detectors: List[FilterType],
    batch_size: int,
    num_beams: int,
    num_return_sequences: int,
    max_length: int,
) -> Dict[str, List]:

    # Initialize the scores dictionary
    records: Dict[str, List] = {f"{detector}": [] for detector in detectors}

    for batch_idx, batch in enumerate(data_loader):
        x = batch["input_ids"]

        output = model.generate(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            max_length=max_length,
            num_beams=num_beams,
            num_return_sequences=num_return_sequences,
            early_stopping=True,
            return_dict_in_generate=True,
            output_scores=True,
            output_hidden_states=True,
        )

        # Should be a dictionary with keys ood scores,
        # each containing a numpy array of shape (batch_size, num_return_sequences))

        ood_scores = evaluate_batch(output, detectors)
        ood_scores = {k: scores.tolist() for k, scores in ood_scores.items()}

        for k, scores in ood_scores.items():
            records[k].extend(scores)

        # A list of list ie each returned sequence for each batch
        decoded_sequences = tokenizer.batch_decode(
            output.sequences, skip_special_tokens=True
        )

        sequences_scores = output.sequences_scores.tolist()
        records["likelyhood"].extend(sequences_scores)

    return records


def try_load_dataset_config(dataset_name: str, dataset_config: str) -> DatasetDict:
    src, tgt = dataset_config.split("-")

    # Try to load the dataset from the datasets library with one config or its permutation
    try:
        dataset = load_dataset(dataset_name, dataset_config)
    except ValueError:
        dataset_config = tgt + "-" + src
        src, tgt = dataset_config.split("-")

        try:
            dataset = load_dataset(
                dataset_name, dataset_config, ignore_verifications=True
            )
        except ValueError:
            raise ValueError(
                "Invalid dataset config. None of the following configs are valid: "
                + dataset_config
                + ", "
                + tgt
                + "-"
                + src
            )

    return dataset
