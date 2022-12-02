from typing import Dict

DATASETS_CONFIGS: Dict[Dict] = {}

BASE_CONFIG = {
    "batch_size": 8,
}

# Datasets configs with english (en) as target language:

# wmt16
DATASETS_CONFIGS["wmt16_de_en"] = BASE_CONFIG | {
    "dataset_name": "wmt16",
    "dataset_config": "de-en",
}
DATASETS_CONFIGS["wmt16_fr_en"] = BASE_CONFIG | {
    "dataset_name": "wmt16",
    "dataset_config": "fr-en",
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


if __name__ == "__main__":
    print("Available datasets configs:")
    for k, v in DATASETS_CONFIGS.items():
        print(k)
