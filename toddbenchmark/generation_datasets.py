from typing import Tuple

from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from .utils_generation import try_load_dataset_config


# Each load_[dataset] function returns a dictionary with the following keys: "train", "validation", "test" containing
# a list of tuples (input, target) for each split.


def print_dataset(name, dataset):
    for k, v in dataset.items():
        print(f"{name} {k} dataset size: {len(v)}")


def no_empty_dataset_sanity_check(name, dataset):
    for split, data in dataset.items():
        # check data length:
        assert len(data) > 0, f"{name} {split} dataset is empty"


def load_daily_dialog(tokenizer, dataset_name):
    dataset = load_dataset(dataset_name)

    def accumulate_dialogs(ds):
        _dataset = []
        for element in ds["dialog"]:
            acc = ""
            for k, sentence in enumerate(element[:-1]):
                acc += sentence + tokenizer.eos_token
                _dataset.append((acc, element[k + 1]))

        return _dataset

    train = accumulate_dialogs(dataset["train"])
    val = accumulate_dialogs(dataset["validation"])
    test = accumulate_dialogs(dataset["test"])

    return {"train": train, "validation": val, "test": test}


def load_multi_woz_v22(tokenizer, dataset_name, dataset_config=None):
    dataset = load_dataset(dataset_name, dataset_config, ignore_verifications=True)

    def accumulate_dialogs(ds):
        _dataset = []
        for element in ds:
            acc = ""
            for k, sentence in enumerate(element["turns"]["utterance"][:-1]):
                acc += sentence + tokenizer.eos_token
                _dataset.append((acc, element["turns"]["utterance"][k + 1]))
        return _dataset

    train = accumulate_dialogs(dataset["train"])
    val = accumulate_dialogs(dataset["validation"])
    test = accumulate_dialogs(dataset["test"])

    return {"train": train, "validation": val, "test": test}


def load_movieqa():
    dataset = load_dataset("wiki_movies", ignore_verifications=True)

    def process_split(ds):
        _dataset = []
        for element in ds:
            q, a = element.split("\t")
            q = q[1:]
            _dataset.append((q, a))
        return _dataset

    train = process_split(dataset["train"]["text"])
    val = process_split(dataset["validation"]["text"])
    test = process_split(dataset["test"]["text"])

    return {"train": train, "validation": val, "test": test}


def load_silicone_dataset(
    dataset_name,
    dataset_config,
):
    dataset = load_dataset(dataset_name, dataset_config, ignore_verifications=True)

    def accumulate_dialogs(ds):
        _dataset = [(e["Utterance"], "a") for e in ds]
        return _dataset

    train = accumulate_dialogs(dataset["train"])
    val = accumulate_dialogs(dataset["validation"])
    test = accumulate_dialogs(dataset["test"])

    return {"train": train, "validation": val, "test": test}


def load_translation_dataset(
    dataset_name,
    dataset_config,
):
    src, tgt = dataset_config.split("-")

    dataset = try_load_dataset_config(dataset_name, dataset_config)

    def process_split(ds):
        _dataset = []
        for element in ds:
            _dataset.append((element[src], element[tgt]))
        return _dataset

    train = process_split(dataset["train"]["translation"])
    val = process_split(dataset["validation"]["translation"])
    test = process_split(dataset["test"]["translation"])

    return {"train": train, "validation": val, "test": test}


def load_tatoeba_dataset(
    dataset_name,
    dataset_config,
):
    datasets = try_load_dataset_config(dataset_name, dataset_config)
    src, tgt = dataset_config.split("-")
    if datasets["test"]["sourceLang"][0] == src:
        source_string = "sourceString"
        target_string = "targetString"
    else:
        source_string = "targetString"
        target_string = "sourceString"

    full_validation = [
        (x[source_string], x[target_string]) for x in datasets["validation"]
    ]
    full_test = [(x[source_string], x[target_string]) for x in datasets["test"]]

    # Use 0.8 of the validation set as the training set
    train = full_validation[: int(0.8 * len(full_validation))]
    validation = full_validation[int(0.8 * len(full_validation)) :]
    test = full_test

    return {"train": train, "validation": validation, "test": test}


def load_wmt16_dataset(
    dataset_name,
    dataset_config,
):
    return load_translation_dataset(dataset_name, dataset_config=dataset_config)


def load_news_commentary_dataset(
    dataset_name,
    dataset_config,
):
    src, tgt = dataset_config.split("-")

    dataset = try_load_dataset_config(dataset_name, dataset_config)

    def process_split(ds):
        _dataset = []
        for element in ds:
            _dataset.append((element[src], element[tgt]))
        return _dataset

    all = process_split(dataset["train"]["translation"])

    # train, val and test sizes:
    train_size = int(len(all) * 0.8)
    val_size = int(len(all) * 0.1)
    test_size = len(all) - train_size - val_size

    train = all[:train_size]
    val = all[train_size : train_size + val_size]
    test = all[train_size + val_size :]

    return {"train": train, "validation": val, "test": test}


# load emea from load_translation_dataset:
def load_emea_dataset(dataset_name, dataset_config):
    src, tgt = dataset_config.split("-")

    dataset = try_load_dataset_config(dataset_name, dataset_config)

    def process_split(ds):
        _dataset = []
        for element in ds:
            _dataset.append((element[src], element[tgt]))
        return _dataset

    all = process_split(dataset["train"]["translation"])

    # train, val and test sizes:
    train_size = int(len(all) * 0.8)
    val_size = int(len(all) * 0.1)
    test_size = len(all) - train_size - val_size

    train = all[:train_size]
    val = all[train_size : train_size + val_size]
    test = all[train_size + val_size :]

    return {"train": train, "validation": val, "test": test}


def load_europarl_dataset(dataset_name, dataset_config):
    lang1, lang2 = dataset_config.split("-")
    switch_lang = False
    try:
        dataset = load_dataset(
            dataset_name, lang1=lang1, lang2=lang2, ignore_verifications=True
        )
    except ValueError:
        try:
            dataset = load_dataset(
                dataset_name, lang1=lang2, lang2=lang1, ignore_verifications=True
            )
            switch_lang = True
        except ValueError:
            raise ValueError(
                "Invalid dataset config. None of the following configs are valid: "
                + dataset_config
                + ", "
                + lang2
                + "-"
                + lang1
            )

    dataset = dataset["train"]

    if switch_lang:
        _dataset = [(d["translation"][lang2], d["translation"][lang1]) for d in dataset]
    else:
        _dataset = [(d["translation"][lang1], d["translation"][lang2]) for d in dataset]

    # split dataset into train, validation and test with 70%, 20% and 10% of the data
    train_size = int(0.7 * len(_dataset))
    val_size = int(0.2 * len(_dataset))
    # test_size = len(_dataset) - train_size - val_size

    train = _dataset[:train_size]
    val = _dataset[train_size : train_size + val_size]
    test = _dataset[train_size + val_size :]

    return {"train": train, "validation": val, "test": test}


def load_amazon_reviews_multi(dataset_name, dataset_config):
    dataset = load_dataset(dataset_name, dataset_config, ignore_verifications=True)

    train = [(d["review_title"], d["review_title"]) for d in dataset["train"]]
    val = [(d["review_title"], d["review_title"]) for d in dataset["validation"]]
    test = [(d["review_title"], d["review_title"]) for d in dataset["test"]]

    return {"train": train, "validation": val, "test": test}


def prep_dataset(
    dataset_name,
    dataset_config,
    tokenizer,
    train_max_size=-1,
    validation_max_size=-1,
    test_max_size=-1,
) -> Tuple[Dataset, Dataset, Dataset]:
    """

    :param dataset_name:
    :param dataset_config:
    :param tokenizer:
    :return: Tuple of (train, validation, test) datasets. Each dataset is a list of dictionaries with keys
    "source", "target"
    """

    if dataset_name == "daily_dialog":
        dataset = load_daily_dialog(
            tokenizer,
            dataset_name,
        )
    elif dataset_name == "Helsinki-NLP/tatoeba_mt":
        dataset = load_tatoeba_dataset(
            dataset_name,
            dataset_config,
        )
    elif dataset_name == "multi_woz_v22":
        dataset = load_multi_woz_v22(
            tokenizer,
            dataset_name,
            dataset_config,
        )

    elif dataset_name == "silicone":
        dataset = load_silicone_dataset(
            dataset_name,
            dataset_config,
        )

    elif dataset_name == "movieqa":
        dataset = load_movieqa()

    elif dataset_name == "wmt16":
        dataset = load_wmt16_dataset(
            dataset_name,
            dataset_config,
        )
    elif dataset_name == "news_commentary":
        dataset = load_news_commentary_dataset(
            dataset_name,
            dataset_config,
        )
    elif dataset_name == "qanastek/EMEA-V3":
        dataset = load_emea_dataset(
            dataset_name,
            dataset_config,
        )

    elif dataset_name == "europarl_bilingual":
        dataset = load_europarl_dataset(
            dataset_name,
            dataset_config,
        )

    elif dataset_name == "amazon_reviews_multi":
        dataset = load_amazon_reviews_multi(
            dataset_name,
            dataset_config,
        )
    else:
        raise ValueError("Dataset not supported")

    def to_dict(dataset, split):
        ds = dataset[split]
        return [{"source": s, "target": t} for s, t in ds]

    train = Dataset.from_list(to_dict(dataset, "train")[:train_max_size])
    val = Dataset.from_list(to_dict(dataset, "validation")[:validation_max_size])
    test = Dataset.from_list(to_dict(dataset, "test")[:test_max_size])

    return train, val, test


def prep_model(model_name):
    if model_name == "microsoft/DialoGPT-medium" or model_name == "tosin/dialogpt_mwoz":
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.truncation_side = "left"
        tokenizer.model_max_length = 50
        model = AutoModelForCausalLM.from_pretrained(model_name)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    return model, tokenizer


def prep_inputs(x, tokenizer, dataset_name):
    if dataset_name == "daily_dialog" or dataset_name == "movieqa":
        inputs = tokenizer(x, return_tensors="pt", truncation=True)
    else:
        inputs = tokenizer(x, return_tensors="pt", truncation=True)

    return inputs


if __name__ == "__main__":
    pass
