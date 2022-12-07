from torch.utils.data import DataLoader

from .classification_datasets import prep_dataset

BASE_CONFIG = {"batch_size": 16}

EN_CONFIGS = {
    "amazon_reviews_multi": {
        "label": 30,
        "metric": "mnli",
        "keys": ("review_body", None),
        "language": "en",
    },
    "go_emotions": {"label": 28, "metric": "mnli", "keys": ("text", None)},
    "sst2": {"label": 2, "metric": "sst2", "keys": ("text", None)},
    "imdb": {"label": 2, "metric": "sst2", "keys": ("text", None)},
    "20ng": {"label": 20, "metric": "mnli", "keys": ("text_scr", None)},
    "trec": {"label": 6, "metric": "mnli", "keys": ("text_scr", None)},
    "mnli": {"label": 3, "metric": "mnli", "keys": ("premise", "hypothesis")},
    "snli": {"label": 3, "metric": "mnli", "keys": ("premise", "hypothesis")},
    "rte": {"label": 2, "metric": "mnli", "keys": ("sentence1", "sentence2")},
    "b77": {"label": 77, "metric": "mnli", "keys": ("text", None)},
    "massive": {"label": 60, "metric": "mnli", "keys": ("text", None)},
    "trec_fine": {"label": 50, "metric": "mnli", "keys": ("text_scr", None)},
    "emotion": {"label": 6, "metric": "mnli", "keys": ("text", None)},
}

FR_CONFIGS = {
    "fr_cls": {"label": 2, "metric": "mnli", "keys": ("text", None)},
    "fr_xnli": {"label": 3, "metric": "mnli", "keys": ("text", None)},
    "fr_pawsx": {"label": 2, "metric": "mnli", "keys": ("text", None)},
    "fr_allocine": {"label": 2, "metric": "mnli", "keys": ("text", None)},
    "fr_xstance": {"label": 2, "metric": "mnli", "keys": ("text", None)},
    "fr_swiss_judgement": {"label": 2, "metric": "mnli", "keys": ("text", None)},
    "fr_tweet_sentiment": {"label": 3, "metric": "mnli", "keys": ("text", None)},
}
DE_CONFIGS = {
    "de_xstance": {"label": 2, "metric": "mnli", "keys": ("text", None)},
    "de_swiss_judgement": {"label": 2, "metric": "mnli", "keys": ("text", None)},
    "de_tweet_sentiment": {"label": 3, "metric": "mnli", "keys": ("text", None)},
    "de_pawsx": {"label": 2, "metric": "mnli", "keys": ("text", None)},
}
ES_CONFIGS = {
    "es_tweet_sentiment": {"label": 3, "metric": "mnli", "keys": ("text", None)},
    "es_pawsx": {"label": 2, "metric": "mnli", "keys": ("text", None)},
    "es_cine": {"label": 5, "metric": "mnli", "keys": ("text", None)},
    "es_tweet_inde": {"label": 3, "metric": "mnli", "keys": ("text", None)},
}

DATASETS_CONFIGS = EN_CONFIGS | FR_CONFIGS | DE_CONFIGS | ES_CONFIGS
DATASETS_CONFIGS = {
    name: BASE_CONFIG | config for name, config in DATASETS_CONFIGS.items()
}


def load_requested_dataset(
    config_name: str,
    tokenizer,
    train_size: int = 3000,
    validation_size: int = 3000,
    test_size: int = 3000,
):
    def tokenize_function(examples):
        return tokenizer(text=examples["text"], truncation=True)

    datasets = {}

    if config_name not in DATASETS_CONFIGS:
        raise ValueError(
            f"Invalid dataset config name: {config_name}. "
            f"Available configs: {list(DATASETS_CONFIGS.keys())}"
        )

    config = DATASETS_CONFIGS[config_name]
    train_dataset, validation_dataset, test_dataset = prep_dataset(
        config_name,
        config,
        tokenizer,
        train_max_size=train_size,
        validation_max_size=validation_size,
        test_max_size=test_size,
    )

    test_dataset = test_dataset.map(
        tokenize_function,
        batched=True,
        num_proc=4,
    )

    validation_dataset = validation_dataset.map(
        tokenize_function,
        batched=True,
        num_proc=4,
    )

    test_dataset = test_dataset.map(
        tokenize_function,
        batched=True,
        num_proc=4,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=4,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=4,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=4,
    )

    return train_loader, validation_loader, test_loader
