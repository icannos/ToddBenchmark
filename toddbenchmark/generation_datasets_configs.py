from typing import Dict

import torch
from torch.utils.data import DataLoader
from .generation_datasets import prep_dataset

DATASETS_CONFIGS: Dict[str, Dict] = {}
TRANSLATION_DATASETS: Dict[str, Dict] = {}

BASE_CONFIG = {
    "batch_size": 8,
}

# Datasets configs with english (en) as target language:

# wmt16
TRANSLATION_DATASETS["wmt16_de_en"] = BASE_CONFIG | {
    "dataset_name": "wmt16",
    "dataset_config": "de-en",
}

# news_commentary
TRANSLATION_DATASETS["news_commentary_en_de"] = BASE_CONFIG | {
    "dataset_name": "news_commentary",
    "dataset_config": "de-en",
}

# Helsinki-NLP TaToeba, german english
TRANSLATION_DATASETS["tatoeba_mt_deu_eng"] = BASE_CONFIG | {
    "dataset_name": "Helsinki-NLP/tatoeba_mt",
    "dataset_config": "deu-eng",
}

# Helsinki-NLP TaToeba, Spanish English
TRANSLATION_DATASETS["tatoeba_mt_spa_eng"] = BASE_CONFIG | {
    "dataset_name": "Helsinki-NLP/tatoeba_mt",
    "dataset_config": "spa-eng",
    "model_config": "es-en",
}

# Helsinki-NLP TaToeba, Catalan English
TRANSLATION_DATASETS["tatoeba_mt_cat_eng"] = BASE_CONFIG | {
    "dataset_name": "Helsinki-NLP/tatoeba_mt",
    "dataset_config": "cat-eng",
}

TRANSLATION_DATASETS["tatoeba_mt_fra_eng"] = BASE_CONFIG | {
    "dataset_name": "Helsinki-NLP/tatoeba_mt",
    "dataset_config": "fra-eng",
}

TRANSLATION_DATASETS["tatoeba_mt_afr_eng"] = BASE_CONFIG | {
    "dataset_name": "Helsinki-NLP/tatoeba_mt",
    "dataset_config": "afr-eng",
}

TRANSLATION_DATASETS["tatoeba_mt_afr_eng"] = BASE_CONFIG | {
    "dataset_name": "Helsinki-NLP/tatoeba_mt",
    "dataset_config": "afr-eng",
}

TRANSLATION_DATASETS["tatoeba_mt_ita_eng"] = BASE_CONFIG | {
    "dataset_name": "Helsinki-NLP/tatoeba_mt",
    "dataset_config": "ita-eng",
}

TRANSLATION_DATASETS["tatoeba_mt_ita_eng"] = BASE_CONFIG | {
    "dataset_name": "Helsinki-NLP/tatoeba_mt",
    "dataset_config": "ita-eng",
}

TRANSLATION_DATASETS["tatoeba_mt_est_eng"] = BASE_CONFIG | {
    "dataset_name": "Helsinki-NLP/tatoeba_mt",
    "dataset_config": "est-eng",
}

TRANSLATION_DATASETS["tatoeba_mt_epo_eng"] = BASE_CONFIG | {
    "dataset_name": "Helsinki-NLP/tatoeba_mt",
    "dataset_config": "epo-eng",
}

TRANSLATION_DATASETS["tatoeba_mt_heb_eng"] = BASE_CONFIG | {
    "dataset_name": "Helsinki-NLP/tatoeba_mt",
    "dataset_config": "heb-eng",
}

TRANSLATION_DATASETS["tatoeba_mt_grc_eng"] = BASE_CONFIG | {
    "dataset_name": "Helsinki-NLP/tatoeba_mt",
    "dataset_config": "grc-eng",
}

TRANSLATION_DATASETS["tatoeba_mt_eus_eng"] = BASE_CONFIG | {
    "dataset_name": "Helsinki-NLP/tatoeba_mt",
    "dataset_config": "eus-eng",
}

TRANSLATION_DATASETS["tatoeba_mt_rus_eng"] = BASE_CONFIG | {
    "dataset_name": "Helsinki-NLP/tatoeba_mt",
    "dataset_config": "rus-eng",
}

TRANSLATION_DATASETS["tatoeba_mt_pol_eng"] = BASE_CONFIG | {
    "dataset_name": "Helsinki-NLP/tatoeba_mt",
    "dataset_config": "pol-eng",
    "model_config": "pl-en",
}

TRANSLATION_DATASETS["tatoeba_mt_nld_eng"] = BASE_CONFIG | {
    "dataset_name": "Helsinki-NLP/tatoeba_mt",
    "dataset_config": "nld-eng",
    "model_config": "nl-en",
}

TRANSLATION_DATASETS["tatoeba_mt_kor_eng"] = BASE_CONFIG | {
    "dataset_name": "Helsinki-NLP/tatoeba_mt",
    "dataset_config": "kor-eng",
}

TRANSLATION_DATASETS["tatoeba_mt_kor_eng"] = BASE_CONFIG | {
    "dataset_name": "Helsinki-NLP/tatoeba_mt",
    "dataset_config": "kor-eng",
}

TRANSLATION_DATASETS["tatoeba_mt_ukr_eng"] = BASE_CONFIG | {
    "dataset_name": "Helsinki-NLP/tatoeba_mt",
    "dataset_config": "ukr-eng",
}

TRANSLATION_DATASETS["tatoeba_mt_urd_eng"] = BASE_CONFIG | {
    "dataset_name": "Helsinki-NLP/tatoeba_mt",
    "dataset_config": "urd-eng",
}

TRANSLATION_DATASETS["tatoeba_mt_urd_eng"] = BASE_CONFIG | {
    "dataset_name": "Helsinki-NLP/tatoeba_mt",
    "dataset_config": "urd-eng",
}

DATASETS_CONFIGS = DATASETS_CONFIGS | TRANSLATION_DATASETS

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

# Summarization

SUMMARIZATION_DATASETS = {}

SUMMARIZATION_DATASETS["cnn_dailymail"] = BASE_CONFIG | {
    "dataset_name": "cnn_dailymail",
    "dataset_config": "3.0.0",
}

SUMMARIZATION_DATASETS["xsum"] = BASE_CONFIG | {
    "dataset_name": "xsum",
    "dataset_config": "",
}

SUMMARIZATION_DATASETS["billsum"] = BASE_CONFIG | {
    "dataset_name": "billsum",
    "dataset_config": "",
}

SUMMARIZATION_DATASETS["multi_news"] = BASE_CONFIG | {
    "dataset_name": "multi_news",
    "dataset_config": "",
}

DATASETS_CONFIGS = DATASETS_CONFIGS | SUMMARIZATION_DATASETS


def load_requested_dataset(
    config_name: str,
    tokenizer,
    batch_size: int = 16,
    train_size: int = 3000,
    validation_size: int = 3000,
    test_size: int = 3000,
    update_input_fn=lambda sample: sample,
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

    train_dataset = train_dataset.map(
        update_input_fn,
    )
    validation_dataset = validation_dataset.map(
        update_input_fn,
    )
    test_dataset = test_dataset.map(update_input_fn)

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
