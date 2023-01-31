from typing import Dict

import torch
from torch.utils.data import DataLoader
from .generation_datasets import prep_dataset

DATASETS_CONFIGS: Dict[str, Dict] = {}

BASE_CONFIG = {
    "batch_size": 8,
}

# Datasets configs with english (en) as target language:

# wmt16
DATASETS_CONFIGS["wmt16_de_en"] = BASE_CONFIG | {
    "dataset_name": "wmt16",
    "dataset_config": "de-en",
}

# news_commentary
DATASETS_CONFIGS["news_commentary_en_de"] = BASE_CONFIG | {
    "dataset_name": "news_commentary",
    "dataset_config": "de-en",
}

# Helsinki-NLP TaToeba, german english
DATASETS_CONFIGS["tatoeba_mt_deu_eng"] = BASE_CONFIG | {
    "dataset_name": "Helsinki-NLP/tatoeba_mt",
    "dataset_config": "deu-eng",
}

# Helsinki-NLP TaToeba, Spanish English
DATASETS_CONFIGS["tatoeba_mt_spa_eng"] = BASE_CONFIG | {
    "dataset_name": "Helsinki-NLP/tatoeba_mt",
    "dataset_config": "spa-eng",
}

# Helsinki-NLP TaToeba, Catalan English
DATASETS_CONFIGS["tatoeba_mt_cat_eng"] = BASE_CONFIG | {
    "dataset_name": "Helsinki-NLP/tatoeba_mt",
    "dataset_config": "cat-eng",
}

###### Question Answering ######

DATASETS_CONFIGS["openbookqa_answerable"] = BASE_CONFIG | {
    "dataset_name": "openbookqa",
    "dataset_config": "answerable",
}

DATASETS_CONFIGS["openbookqa_unanswerable"] = BASE_CONFIG | {
    "dataset_name": "openbookqa",
    "dataset_config": "unanswerable",
}

DATASETS_CONFIGS["ai2arcchallenge_answerable"] = BASE_CONFIG | {
    "dataset_name": "ai2_arc",
    "dataset_config": "answerable_ARC-Challenge",
}

DATASETS_CONFIGS["ai2arceasy_answerable"] = BASE_CONFIG | {
    "dataset_name": "ai2_arc",
    "dataset_config": "answerable_ARC-Easy",
}

DATASETS_CONFIGS["ai2arcchallenge_unanswerable"] = BASE_CONFIG | {
    "dataset_name": "ai2_arc",
    "dataset_config": "unanswerable_ARC-Challenge",
}

DATASETS_CONFIGS["ai2arceasy_unanswerable"] = BASE_CONFIG | {
    "dataset_name": "ai2_arc",
    "dataset_config": "unanswerable_ARC-Easy",
}


DATASETS_CONFIGS["sciq_answerable"] = BASE_CONFIG | {
    "dataset_name": "sciq",

    "dataset_config": "answerable",
}

DATASETS_CONFIGS["sciq_unanswerable"] = BASE_CONFIG | {
    "dataset_name": "sciq",
    "dataset_config": "unanswerable",
}

DATASETS_CONFIGS["tweetqa_answerable"] = BASE_CONFIG | {
    "dataset_name": "tweetqa",
    "dataset_config": "answerable",
}

DATASETS_CONFIGS["tweetqa_unanswerable"] = BASE_CONFIG | {
    "dataset_name": "tweetqa",
    "dataset_config": "unanswerable",
}

DATASETS_CONFIGS["quartz_answerable"] = BASE_CONFIG | {
    "dataset_name": "quartz",
    "dataset_config": "answerable",
}

DATASETS_CONFIGS["quartz_unanswerable"] = BASE_CONFIG | {
    "dataset_name": "quartz",
    "dataset_config": "unanswerable",
}

DATASETS_CONFIGS["squad_v2_answerable"] = BASE_CONFIG | {
    "dataset_name": "squad_v2",
    "dataset_config": "answerable",
}

DATASETS_CONFIGS["squad_v2_unanswerable"] = BASE_CONFIG | {
    "dataset_name": "squad_v2",
    "dataset_config": "unanswerable",
}


# DATASETS_CONFIGS["cuad_answerable"] = BASE_CONFIG | {
#     "dataset_name": "cuad",
#     "dataset_config": "answerable",
# }
#
# DATASETS_CONFIGS["cuad_unanswerable"] = BASE_CONFIG | {
#     "dataset_name": "cuad",
#     "dataset_config": "unanswerable",
# }


def load_requested_dataset(
    config_name: str,
    tokenizer,
    batch_size: int = 16,
    train_size: int = 3000,
    validation_size: int = 3000,
    test_size: int = 3000,
):

    datasets = {}

    if config_name not in DATASETS_CONFIGS:
        raise ValueError(
            f"Invalid dataset config name: {config_name}. "
            f"Available configs: {list(DATASETS_CONFIGS.keys())}"
        )

    config = DATASETS_CONFIGS[config_name]
    train_dataset, validation_dataset, test_dataset = prep_dataset(
        config["dataset_name"],
        config["dataset_config"],
        tokenizer,
        train_max_size=train_size,
        validation_max_size=validation_size,
        test_max_size=test_size,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
    )

    return train_loader, validation_loader, test_loader


if __name__ == "__main__":
    print("Available datasets configs:")
    for k, v in DATASETS_CONFIGS.items():
        print(k)
        print(v)
        print("===========")
