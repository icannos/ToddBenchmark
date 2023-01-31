from typing import Dict
import os

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


def load_requested_dataset(
    config_name: str,
    tokenizer,
    batch_size: int = 16,
    train_size: int = 3000,
    validation_size: int = 3000,
    test_size: int = 3000,
):

    if os.path.isfile(config_name) and config_name.endswith(".csv"):
        train_dataset, validation_dataset, test_dataset = prep_dataset(
            config_name,
            train_max_size=train_size,
            validation_max_size=validation_size,
            test_max_size=test_size,
        )

    elif config_name in DATASETS_CONFIGS:

        config = DATASETS_CONFIGS[config_name]
        train_dataset, validation_dataset, test_dataset = prep_dataset(
            config["dataset_name"],
            config["dataset_config"],
            tokenizer,
            train_max_size=train_size,
            validation_max_size=validation_size,
            test_max_size=test_size,
        )
    else:
        raise ValueError(
            f"Invalid dataset config name: {config_name}. "
            f"Available configs: {list(DATASETS_CONFIGS.keys())}"
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
