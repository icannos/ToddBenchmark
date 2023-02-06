from collections import defaultdict
from typing import List, Dict, Any, Optional, Callable

import torch
from datasets import load_dataset, DatasetDict
from torch.utils.data import DataLoader
import evaluate

from Todd import ScorerType


def prepare_detectors(
        detectors: List[ScorerType], model, loader: DataLoader, tokenizer
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
            batch["text"], padding=True, truncation=True, return_tensors="pt"
        )

        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        output = model(
            inputs["input_ids"], attention_mask=inputs["attention_mask"], output_hidden_states=True
        )
        output['encoder_hidden_states'] = output['hidden_states']

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

def evaluate_dataloader(
        model,
        data_loader: DataLoader,
        tokenizer,
        detectors: List[ScorerType],
) -> Dict[str, List]:
    # Initialize the scores dictionary
    records: Dict[str, List] = {
        f"{detector}+{score_name}": []
        for detector in detectors
        for score_name in detector.score_names
    }

    records["likelihood"] = []
    records["pred_label"] = []
    records["true_label"] = []
    records["correct"] = []

    for batch_idx, batch in enumerate(data_loader):

        inputs = tokenizer(
            batch["text"], padding=True, truncation=True, return_tensors="pt"
        )
        labels = batch["labels"]

        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        output = model(inputs["input_ids"], attention_mask=inputs["attention_mask"], output_hidden_states=True)
        output['encoder_hidden_states'] = output['hidden_states']

        # Should be a dictionary with keys ood scores,
        # each containing a numpy array of shape (batch_size, num_return_sequences))

        ood_scores = evaluate_batch(output, detectors)

        ood_scores = {
            k: (scores.tolist() if not isinstance(scores, list) else scores)
            for k, scores in ood_scores.items()
        }

        for k, scores in ood_scores.items():
            records[k].extend(scores)

        logits = output.logits
        likelihood = torch.nn.functional.log_softmax(logits, dim=-1)

        pred_labels = torch.argmax(likelihood, dim=-1)

        records["pred_label"].extend(pred_labels.tolist())
        records["true_label"].extend(labels.tolist())
        records["correct"].extend((pred_labels == labels).tolist())

    return records
