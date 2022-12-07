BASE_CONFIG = {}


EN_CONFIGS = {
    "amazon_reviews_multi": {
        "label": 30,
        "metric": "mnli",
        "keys": ("review_body", None),
    },
    "go_emotions": {"label": 28, "metric": "mnli", "keys": ("text", None)},
    "sst2": {"label": 2, "metric": "sst2", "keys": ("sentence", None)},
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

ALL_CONFIGS = EN_CONFIGS | FR_CONFIGS | DE_CONFIGS | ES_CONFIGS
