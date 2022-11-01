from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM

# Each load_[dataset] function returns a dictionary with the following keys: "train", "validation", "test" containing
# a list of tuples (input, target) for each split.


def print_dataset(name, dataset):
    for k, v in dataset.items():
        print(f"{name} {k} dataset size: {len(v)}")


def no_empty_dataset_sanity_check(name, dataset):
    for split, data in dataset.items():
        # check data length:
        assert len(data) > 0, f"{name} {split} dataset is empty"


class GenerationDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        return self.dataset[index]

    def map(self, fn):
        self.dataset = [x | fn(x) for x in self.dataset]
        return self

    def __len__(self):
        return len(self.dataset)

    def __repr__(self):
        return f"GenerationDataset(len={len(self)}, features={self.dataset[0].keys()})"


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

    def process_split(ds):
        _dataset = []
        for element in ds:
            _dataset.append((element[src], element[tgt]))
        return _dataset

    train = process_split(dataset["train"]["translation"])
    val = process_split(dataset["validation"]["translation"])
    test = process_split(dataset["test"]["translation"])

    return {"train": train, "validation": val, "test": test}


def load_wmt16_dataset(
    dataset_name,
    dataset_config,
):
    return load_translation_dataset(dataset_name, dataset_config=dataset_config)


def load_news_commentary_dataset(
    dataset_name,
    dataset_config,
):
    return load_translation_dataset(
        dataset_name,
        dataset_config=dataset_config,
    )


# load emea from load_translation_dataset:
def load_emea_dataset(dataset_name, dataset_config):
    return load_translation_dataset(
        dataset_name,
        dataset_config=dataset_config,
    )


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
):
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
        dataset = None

    def to_dict(dataset, split):
        ds = dataset[split]
        return [{"source": s, "target": t} for s, t in ds]

    train = GenerationDataset(to_dict(dataset, "train"))
    val = GenerationDataset(to_dict(dataset, "validation"))
    test = GenerationDataset(to_dict(dataset, "test"))

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
