from collections import defaultdict, Counter
from math import log
from typing import List, Dict, Any, Optional, Callable
from tqdm import tqdm

import torch
from datasets import load_dataset, DatasetDict
from transformers.generation import BeamSearchEncoderDecoderOutput
from torch.utils.data import DataLoader

from Todd import ScorerType
from Todd.query_based_scorers import QueryBasedScorer
from Todd.classifier_scorer import ClassifierScorer


def fit_models(
        tokenizer,
        model,
        loader: DataLoader,
        **kwargs
):

    probs = []
    for batch in tqdm(loader):
        inputs = tokenizer(
            batch["source"], padding=True, truncation=True, return_tensors="pt"
        ).to(model.device)
        output = model.generate(input_ids=inputs["input_ids"],
                                attention_mask=inputs["attention_mask"],
                                max_new_tokens=kwargs.get("max_length", 32),
                                return_dict_in_generate=True,
                                output_scores=True,
                                )
        # torch.equal(output.sequences[:, :inputs.input_ids.shape[-1]], inputs.input_ids):
        scores = output.scores
        probs.append(torch.cat(scores, dim=0).softmax(dim=1).mean(dim=0))
    probs = torch.stack(probs).mean(dim=0)
    return probs


def prepare_detectors_out(
        detectors: List[ScorerType],
        model,
        loader: DataLoader,
        tokenizer,
        **kwargs
) -> List[ScorerType]:
    with torch.no_grad():
        for batch in tqdm(loader):

            inputs = tokenizer(
                batch["source"], padding=True, truncation=True, return_tensors="pt"
            ).to(model.device)

            output = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=kwargs.get("max_length", 32),
                num_beams=kwargs.get("num_beams", 4),
                num_return_sequences=kwargs.get("num_return_sequences", 4),
                early_stopping=True,
                return_dict_in_generate=True,
                output_scores=True,
                output_hidden_states=True,
            )
            # output = BeamSearchEncoderDecoderOutput({k: v.to("cpu") if isinstance(v, torch.Tensor) else tuple(
            #     v_element.to("cpu") if isinstance(v_element, torch.Tensor) else tuple(
            #         v_elem.to("cpu") for v_elem in v_element) for v_element in v) for k, v in output.items()})

            for detector in detectors:
                if isinstance(detector, ClassifierScorer):
                    detector.accumulate(output, [1])

    for detector in detectors:
        if isinstance(detector, ClassifierScorer):
            detector.fit()

    return detectors


def prepare_detectors(
        detectors: List[ScorerType],
        model,
        loader: DataLoader,
        tokenizer,
        **kwargs
) -> List[ScorerType]:
    """
    Fit the detectors on the behavior of the model on the (in) validation set
    :param detectors: List of detectors to fit
    :param model: Model to evaluate
    :param loader: Dataloader (reference set) to evaluate the model on
    :return: List of fitted detectors
    """

    with torch.no_grad():
        for batch in tqdm(loader):

            inputs = tokenizer(
                batch["source"], padding=True, truncation=True, return_tensors="pt"
            ).to(model.device)
            labels = tokenizer(
                batch["target"], padding=True, truncation=True, return_tensors="pt"
            ).to(model.device)

            output = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=kwargs.get("max_length", 32),
                num_beams=kwargs.get("num_beams", 4),
                num_return_sequences=kwargs.get("num_return_sequences", 4),
                early_stopping=True,
                return_dict_in_generate=True,
                output_scores=True,
                output_hidden_states=True,
            )
            # output = BeamSearchEncoderDecoderOutput({k: v.to("cpu") if isinstance(v, torch.Tensor) else tuple(
            #     v_element.to("cpu") if isinstance(v_element, torch.Tensor) else tuple(
            #         v_elem.to("cpu") for v_elem in v_element) for v_element in v) for k, v in output.items()})

            for detector in detectors:
                if isinstance(detector, QueryBasedScorer):
                    # TODO: remove - this is just for debugging
                    # print(tokenizer.batch_decode(output.sequences, skip_special_tokens=True, clean_up_tokenization_spaces=True))

                    sentence_pairs = list(zip(batch["source"],
                                              [tokenizer.decode(output.sequences[i], skip_special_tokens=True) for i in
                                               range(0, len(output.sequences), kwargs.get("num_return_sequences", 4))]))
                    detector.accumulate(sentence_pairs if detector.concat_output else batch["source"], model, tokenizer)
                elif isinstance(detector, ClassifierScorer):
                    detector.accumulate(output, [0])
                else:
                    detector.accumulate(output)

    for detector in detectors:
        if not isinstance(detector, ClassifierScorer):  # Don't fit it until it has OOD data
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
        if not isinstance(detector, QueryBasedScorer):
            s = detector.compute_scores_benchmark(output)
            scores |= {f"{detector}+{k}": v for k, v in flatten_dict(s).items()}
    return scores


def evaluate_dataloader(
        model,
        data_loader: DataLoader,
        tokenizer,
        detectors: List[ScorerType],
        num_beams: int,
        num_return_sequences: int,
        max_length: int,
        metric_eval: Optional[Callable] = None,
) -> Dict[str, List]:

    # Initialize the scores dictionary
    records: Dict[str, List] = {
        f"{detector}+{score_name}": []
        for detector in detectors
        for score_name in (detector.score_names if len(detector.score_names) > 0 else ["score"])
    }

    # print(records)
    records["likelihood"] = []

    for batch_idx, batch in enumerate(tqdm(data_loader)):

        inputs = tokenizer(
            batch["source"], padding=True, truncation=True, return_tensors="pt"
        ).to(model.device)
        # labels = tokenizer(
        #     batch["target"], padding=True, truncation=True, return_tensors="pt"
        # ).to(model.device)

        with torch.no_grad():
            output = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=max_length,
                num_beams=num_beams,
                num_return_sequences=num_return_sequences,
                num_beam_groups=1,
                early_stopping=True,
                return_dict_in_generate=True,
                output_scores=True,
                output_hidden_states=True,
            )

            # inputs = {k: v.to(model.device) for k, v in inputs.items()}
            sentence_pairs = list(zip(batch["source"],
                                      [tokenizer.decode(output.sequences[i], skip_special_tokens=True) for i in
                                       range(0, len(output.sequences), num_return_sequences)]))
            for detector in detectors:
                if isinstance(detector, QueryBasedScorer):
                    records[f"{detector}+score"].extend(detector.score_sentences(sentence_pairs if detector.concat_output else batch["source"], model, tokenizer))

            # output = BeamSearchEncoderDecoderOutput({k: v.to("cpu") if isinstance(v, torch.Tensor) else tuple(
            #     v_element.to("cpu") if isinstance(v_element, torch.Tensor) else tuple(
            #         v_elem.to("cpu") for v_elem in v_element) for v_element in v) for k, v in output.items()})
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
        ).cpu()

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

        for k, v in global_perfs_scores.items():
            if k not in records:
                records[k] = []
            records[k].extend(v)

        if "sequences_scores" in output:
            sequences_scores = output.sequences_scores.view(
                output.sequences_scores.shape[0] // num_return_sequences,
                num_return_sequences,
            ).tolist()
        else:
            sequences_scores = [0.0] * len(batch)  # TODO: fix this

        records["likelihood"].extend(sequences_scores)
    for metric in list(records.keys()):
        if "QueryScorer" in metric:
            keys = [list(x.keys())[0] for x in records[metric][0]]
            swapped_dict = {k: [x[i][k] for x in records[metric]] for i, k in enumerate(keys)}
            records[metric] = swapped_dict
    # print(records)

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
