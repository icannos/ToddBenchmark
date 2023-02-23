import argparse
import json
from pathlib import Path

import numpy as np
from fif import FiForest
from toddbenchmark.utils_generation import mk_file_name
from tqdm import tqdm

AGGREGATIONS = {
    "mean": np.mean,
    "std": np.std,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--in_ds", type=str)
    parser.add_argument("--base_path", type=Path, default="output/")

    return parser.parse_args()


def convert_list(list_ts):
    n_samples = len(list_ts)
    dim = np.zeros(n_samples, dtype=float)

    for i in range(n_samples):
        dim[i] = len(list_ts[i])
    max_dim = int(np.max(dim))

    matrix_ts = np.zeros((n_samples, max_dim))

    for i in range(n_samples):
        matrix_ts[i, : int(dim[i])] = list_ts[i]
    return matrix_ts, dim, max_dim


def aggregate_per_token_scores(json_data, AGGREGATIONS=None):
    for k, v in json_data.items():
        if "token" in k:
            # shape: [N_samples, N_seqs, N_tokens]
            N_samples = len(v)
            N_seqs = len(v[0])

            agg_scores = {}
            for agg_name, agg_func in AGGREGATIONS.items():
                agg_scores[agg_name] = []
                for i in range(N_samples):
                    agg_scores[agg_name].append([])
                    for j in range(N_seqs):
                        agg_scores[agg_name][i].append(agg_func(v[i][j]))

            for agg_name, agg_scores in agg_scores.items():
                json_data["agg"][f"{k}_{agg_name}"] = agg_scores

    return json_data


def compute_fifs(in_json_ref, in_json, score):

    in_ref = [sentences[0] for sentences in in_json_ref[score]]

    array_ts, size_ts, max_size_ts = convert_list(in_ref)
    F = FiForest(array_ts, size_ts, sample_size=32, alpha=0.9, dic_number=1)

    N_samples = len(in_json[score])
    N_seqs = len(in_json[score][0])

    all_sentences = []
    for sentences in in_json[score]:
        all_sentences.extend(sentences)

    array_ts, size_ts, max_size_ts = convert_list(all_sentences)
    fifs = F.compute_paths(array_ts, size_ts)

    fifs = fifs.reshape(
        (
            N_samples,
            N_seqs,
        )
    )

    return fifs.tolist()


def main():
    args = parse_args()

    in_ds = args.in_ds
    base_path = args.base_path

    in_training_scores_path = (
        Path(base_path) / "validation_scores" / mk_file_name(args.model, in_ds, in_ds)
    )
    in_test_scores_path = (
        Path(base_path) / "test_scores" / mk_file_name(args.model, in_ds, in_ds)
    )

    with open(in_training_scores_path, "r") as f:
        in_training_scores = json.load(f)
        if "agg" not in in_training_scores:
            in_training_scores["agg"] = {}

    with open(in_test_scores_path, "r") as f:
        in_test_scores = json.load(f)
        if "agg" not in in_test_scores:
            in_test_scores["agg"] = {}

    for per_token_scores in tqdm(in_training_scores.keys()):
        if "mode=token" not in per_token_scores:
            continue

        fif_in = compute_fifs(in_training_scores, in_test_scores, per_token_scores)

        in_test_scores["agg"][f"fif_{per_token_scores}"] = fif_in

        in_test_scores = aggregate_per_token_scores(in_test_scores, AGGREGATIONS)

    with open(in_test_scores_path, "w") as f:
        json.dump(in_test_scores, f)


if __name__ == "__main__":
    main()
