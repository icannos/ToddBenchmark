import random
from typing import Optional, Dict, Tuple

from tqdm import tqdm

from datasets import load_dataset, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def prep_dataset(
    config_name,
    config,
    tokenizer,
    train_max_size=-1,
    validation_max_size=-1,
    test_max_size=-1,
) -> Tuple[Dataset, Dataset, Dataset]:
    sentence1_key, sentence2_key = config["keys"]

    datasets = None

    if config_name in ("mnli", "rte"):
        datasets = load_glue(config_name)
    elif config_name in ("snli",):
        datasets = load_snli()
    elif config_name == "tweet_eval":
        datasets = load_tweet_eval()
    elif config_name == "amazon_reviews_multi":
        datasets = load_amazon_reviews_multi(config["language"])
    elif config_name == "go_emotions":
        datasets = load_go_emotions()
    elif config_name == "sst2":
        datasets = load_sst2()
    elif config_name == "20ng":
        datasets = load_20ng()
    elif config_name == "trec":
        datasets = load_trec()
    elif config_name == "trec_fine":
        datasets = load_trec(labels="label-fine")
    elif config_name == "imdb":
        datasets = load_imdb()
    elif config_name == "yelp":
        datasets = load_yelp()
    elif config_name == "b77":
        datasets = load_b77()
    elif config_name == "massive":
        datasets = load_massive()
    elif config_name == "emotion":
        datasets = load_emotion()
    elif config_name == "twitterfin":
        datasets = load_twitterfin()
    elif config_name in ("fr_xnli", "fr_pawsx", "fr_cls"):
        datasets = load_flue(config_name)
    elif config_name == "fr_book_reviews":
        datasets = load_fr_book_reviews()
    elif config_name == "fr_allocine":
        datasets = load_fr_allocine()
    elif config_name in ("fr_xstance", "es_xstance", "de_xstance"):
        datasets = load_xstance(config_name)
    elif config_name in ("fr_swiss_judgement", "de_swiss_judgement"):
        datasets = load_swiss_judgement(config_name)
    elif config_name in (
        "fr_tweet_sentiment",
        "es_tweet_sentiment",
        "de_tweet_sentiment",
    ):
        datasets = load_tweet_multil_sentiments(config_name)
    elif config_name in ("fr_pawsx", "es_pawsx", "de_pawsx"):
        datasets = load_pawsx(config_name)
    elif config_name == "de_lexarg":
        datasets = load_german_arg_mining(config_name)
    elif config_name == "es_tweet_inde":
        datasets = load_twitter_catinde(config_name)
    elif config_name == "es_cine":
        datasets = load_muchocine(config_name)
    elif config_name == "es_sst2":
        datasets = load_sst2_es(config_name)
    else:
        raise ValueError(f"Unknown dataset {config_name}")

    def preprocess_function(examples):
        result = {}
        inputs = (
            examples[sentence1_key]
            if sentence2_key is None
            else examples[sentence1_key] + " " + examples[sentence2_key]
        )

        result["text"] = inputs

        result["labels"] = examples["label"] if "label" in examples else 0

        return result

    train_dataset = (
        list(map(preprocess_function, datasets["train"]))
        if "train" in datasets
        else None
    )
    dev_dataset = (
        list(map(preprocess_function, datasets["validation"]))
        if "validation" in datasets
        else None
    )
    test_dataset = (
        list(map(preprocess_function, datasets["test"])) if "test" in datasets else None
    )

    train_dataset = Dataset.from_list(train_dataset[:train_max_size])
    dev_dataset = Dataset.from_list(dev_dataset[:validation_max_size])
    test_dataset = Dataset.from_list(test_dataset[:test_max_size])

    return train_dataset, dev_dataset, test_dataset


def load_swiss_judgement(task_name):
    lang = task_name.split("_")[0]
    datasets = load_dataset("swiss_judgment_prediction", lang)

    train = [x for x in datasets["train"]]
    validation = [x for x in datasets["validation"]]
    test = [x for x in datasets["test"]]

    return {"train": train, "validation": validation, "test": test}


def load_twitter_catinde(task_name):
    datasets = load_dataset("catalonia_independence", "spanish")

    train = [{"text": x["TWEET"], "label": x["LABEL"]} for x in datasets["train"]]
    validation = [
        {"text": x["TWEET"], "label": x["LABEL"]} for x in datasets["validation"]
    ]
    test = [{"text": x["TWEET"], "label": x["LABEL"]} for x in datasets["test"]]

    return {"train": train, "validation": validation, "test": test}


def load_muchocine(task_name):
    datasets = load_dataset("muchocine")
    dataset = [
        {"text": s["review_summary"], "label": int(s["star_rating"])}
        for s in datasets["train"]
    ]

    train = dataset[:2000] + dataset[:2000] + dataset[:2000]
    validation = dataset[1000:2500]
    test = dataset[2500:]

    return {"train": train, "validation": validation, "test": test}


def load_pawsx(task_name):
    lang = task_name.split("_")[0]
    datasets = load_dataset("paws-x", lang)

    def mk_sample(sample):
        return {
            "text": sample["sentence1"] + " " + sample["sentence2"],
            "label": sample["label"],
        }

    datasets = datasets.map(mk_sample, remove_columns=["sentence1", "sentence2"])

    train = [x for x in datasets["train"]]
    validation = [x for x in datasets["validation"]]
    test = [x for x in datasets["test"]]

    return {"train": train, "validation": validation, "test": test}


def load_sst2_es(task_name=None):
    datasets = load_dataset("mrm8488/sst2-es-mt")

    def mk_sample(x):
        return {"text": x["sentence_es"], "label": x["label"]}

    datasets = datasets.map(mk_sample, remove_columns=["sentence_es", "label"])

    train = [x for x in datasets["train"]]
    validation = [x for x in datasets["validation"]]
    test = [x for x in datasets["test"]]

    return {"train": train, "validation": validation, "test": test}


def load_tweet_multil_sentiments(task_name):
    lang = task_name.split("_")[0]
    lang_map = {"en": "english", "es": "spanish", "fr": "french", "de": "german"}
    datasets = load_dataset("cardiffnlp/tweet_sentiment_multilingual", lang_map[lang])

    train = [x for x in datasets["train"]]
    validation = [x for x in datasets["validation"]]
    test = [x for x in datasets["test"]]

    return {"train": train, "validation": validation, "test": test}


def load_german_arg_mining(task_name):
    dataset = load_dataset("joelito/german_argument_mining", ignore_verifications=True)

    label_map = {"conclusion": 0, "definition": 1, "subsumption": 2, "other": 3}
    train = [
        {"text": x["input_sentence"], "label": label_map[x["label"]]}
        for x in dataset["train"]
    ]
    validation = [
        {"text": x["input_sentence"], "label": label_map[x["label"]]}
        for x in dataset["validation"]
    ]
    test = [
        {"text": x["input_sentence"], "label": label_map[x["label"]]}
        for x in dataset["test"]
    ]

    return {"train": train, "validation": validation + train[:2000], "test": test}


def load_xstance(task_name):
    lang = task_name.split("_")[0]

    datasets = load_dataset("strombergnlp/x-stance", lang)

    def mk_sample(sample):
        return {
            "text": sample["question"] + " " + sample["comment"],
            "label": sample["label"],
        }

    datasets = datasets.map(mk_sample)

    train = [x for x in datasets["train"]]
    validation = [x for x in datasets["validation"]] + train[:2000]
    test = [x for x in datasets["test"]]

    return {"train": train, "validation": validation, "test": test}


def load_flue(task_name):
    task_map = {
        "fr_xnli": "XNLI",
        "fr_cls": "CLS",
        "fr_pawsx": "PAWS-X",
    }

    if task_name == "fr_pawsx":
        datasets = load_dataset("flue", task_map[task_name])
        ds = {}

        def mk_sample(x):
            return {
                "text": x["sentence1"] + " " + x["sentence2"],
                "label": x["label"],
            }

        datasets = datasets.map(mk_sample, remove_columns=["sentence1", "sentence2"])
        ds["train"] = [x for x in datasets["train"]][:30000]
        ds["validation"] = [x for x in datasets["validation"]] + [
            x for x in datasets["test"]
        ][30000 : 30000 + 2000]
        ds["test"] = [x for x in datasets["test"]]

    elif task_name == "fr_xnli":

        datasets = load_dataset("flue", "XNLI")

        def mk_sample(x):
            return {
                "text": x["premise"] + " " + x["hypo"],
                "label": x["label"],
            }

        ds = {}
        datasets = datasets.map(mk_sample, remove_columns=["premise", "hypo"])
        ds["train"] = [x for x in datasets["train"]][:50000]
        ds["validation"] = [x for x in datasets["validation"]] + [
            x for x in datasets["test"]
        ][50000 : 50000 + 2000]
        ds["test"] = [x for x in datasets["test"]]

    elif task_name == "fr_cls":
        datasets = load_dataset("flue", "CLS")
        ds = {}
        ds["train"] = [x for x in datasets["train"]]
        ds["validation"] = [x for x in datasets["test"]][:2000] + [
            x for x in datasets["train"]
        ][:2000]
        ds["test"] = [x for x in datasets["test"]][:2000]

    else:
        raise ValueError("Unknown task {}".format(task_name))

    return ds


def load_emotion():
    dataset = load_dataset("emotion")
    dd = {
        "train": [x for x in dataset["train"]],
        "validation": [x for x in dataset["validation"]],
        "test": [x for x in dataset["test"]],
    }

    return dd


def load_b77():
    datasets = load_dataset("banking77")  # label = json
    datasets = {
        "train": datasets["train"],
        "validation": datasets["test"],
        "test": datasets["test"],
    }
    new_datasets = {"train": [], "validation": [], "test": []}
    for split_name in new_datasets.keys():
        arr = datasets[split_name]["label"]
        texts = datasets[split_name]["text"]

        for i in tqdm(range(len(arr))):
            new_datasets[split_name].append(
                {
                    "label": arr[i],
                    "text": texts[i],
                }
            )
    return new_datasets


def load_twitterfin():
    datasets = load_dataset("zeroshot/twitter-financial-news-sentiment")

    _train = [x for x in datasets["train"]]
    _val = [x for x in datasets["validation"]]

    train = _train[:6000]
    val = _train[6000:]
    test = _val

    return {
        "train": train,
        "validation": val,
        "test": test,
    }


def load_massive(lang="en-US"):
    datasets = load_dataset("AmazonScience/massive", lang)

    train_dataset = [
        {"text": x["utt"], "label": x["intent"]} for x in datasets["train"]
    ]
    dev_dataset = [
        {"text": x["utt"], "label": x["intent"]} for x in datasets["validation"]
    ]
    test_dataset = [{"text": x["utt"], "label": x["intent"]} for x in datasets["test"]]

    return {"train": train_dataset, "validation": dev_dataset, "test": test_dataset}


def load_glue(task):
    datasets = load_dataset("glue", task)
    if task == "mnli":
        test_dataset = [d for d in datasets["test_matched"]] + [
            d for d in datasets["test_mismatched"]
        ]
        datasets["test"] = test_dataset

    if task == "rte":
        datasets = {
            "validation": list(datasets["validation"]) + list(datasets["train"])[:2000],
            "test": list(datasets["test"]),
            "train": list(datasets["train"]),
        }
    if task == "sst2":
        datasets = {
            "validation": list(datasets["validation"]) + list(datasets["train"])[:2000],
            "test": list(datasets["test"]),
            "train": list(datasets["train"]),
        }
    return datasets


def load_snli():
    datasets = load_dataset("snli")
    return datasets


def load_20ng():
    all_subsets = (
        "18828_alt.atheism",
        "18828_comp.graphics",
        "18828_comp.os.ms-windows.misc",
        "18828_comp.sys.ibm.pc.hardware",
        "18828_comp.sys.mac.hardware",
        "18828_comp.windows.x",
        "18828_misc.forsale",
        "18828_rec.autos",
        "18828_rec.motorcycles",
        "18828_rec.sport.baseball",
        "18828_rec.sport.hockey",
        "18828_sci.crypt",
        "18828_sci.electronics",
        "18828_sci.med",
        "18828_sci.space",
        "18828_soc.religion.christian",
        "18828_talk.politics.guns",
        "18828_talk.politics.mideast",
        "18828_talk.politics.misc",
        "18828_talk.religion.misc",
    )
    train_dataset = []
    dev_dataset = []
    test_dataset = []
    for i, subset in enumerate(all_subsets):
        dataset = load_dataset("newsgroup", subset)["train"]
        examples = [{"text_scr": d["text"], "label": i} for d in dataset]
        random.shuffle(examples)
        num_train = int(0.8 * len(examples))
        num_dev = int(0.1 * len(examples))
        train_dataset += examples[:num_train]
        dev_dataset += examples[num_train : num_train + num_dev]
        test_dataset += examples[num_train + num_dev :]
    datasets = {"train": train_dataset, "validation": dev_dataset, "test": test_dataset}
    return datasets


def load_fr_book_reviews():
    datasets = load_dataset("Abirate/french_book_reviews", ignore_verifications=True)
    dd = [
        {"text": x["reader_review"], "label": int(x["label"] + 1)}
        for x in datasets["train"]
    ]
    train = dd[:5000]
    val = dd[4000:7000]
    test = dd[7000:]

    return {"train": train, "validation": val, "test": test}


def load_fr_allocine():
    datasets = load_dataset("allocine")
    train = [{"text": x["review"], "label": x["label"]} for x in datasets["train"]][
        :50000
    ]
    val = [{"text": x["review"], "label": x["label"]} for x in datasets["validation"]][
        :5000
    ]
    test = [{"text": x["review"], "label": x["label"]} for x in datasets["test"]][
        :20000
    ]

    return {"train": train, "validation": val, "test": test}


def load_trec(labels="label-coarse"):  # or fine-label
    datasets = load_dataset("trec")
    train_dataset = datasets["train"]
    test_dataset = datasets["test"]
    idxs = list(range(len(train_dataset)))
    random.shuffle(idxs)
    num_reserve = int(len(train_dataset) * 0.1)
    dev_dataset = [
        {
            "text_scr": train_dataset[i]["text"],
            "label": train_dataset[i][labels],
        }
        for i in idxs[-num_reserve:]
    ]
    train_dataset = [
        {
            "text_scr": train_dataset[i]["text"],
            "label": train_dataset[i][labels],
        }
        for i in idxs[:-num_reserve]
    ]
    test_dataset = [{"text_scr": d["text"], "label": d[labels]} for d in test_dataset]
    datasets = {"train": train_dataset, "validation": dev_dataset, "test": test_dataset}
    return datasets


def load_yelp():
    datasets = load_dataset("yelp_polarity")
    train_dataset = datasets["train"]
    idxs = list(range(len(train_dataset) // 10))
    random.shuffle(idxs)
    num_reserve = int(len(train_dataset) // 10 * 0.1)
    dev_dataset = [
        {"text": train_dataset[i]["text"], "label": train_dataset[i]["label"]}
        for i in idxs[-num_reserve:]
    ]
    train_dataset = [
        {"text": train_dataset[i]["text"], "label": train_dataset[i]["label"]}
        for i in idxs[:-num_reserve]
    ]
    test_dataset = datasets["test"]
    datasets = {"train": train_dataset, "validation": dev_dataset, "test": test_dataset}
    return datasets


def load_imdb():
    datasets = load_dataset("imdb", ignore_verifications=True)  # /plain_text')
    train_dataset = datasets["train"]
    unsup_dataset = datasets["unsupervised"]
    idxs = list(range(len(train_dataset)))
    random.shuffle(idxs)
    num_reserve = int(len(train_dataset) * 0.1)
    dev_dataset = [
        {"text": train_dataset[i]["text"], "label": train_dataset[i]["label"]}
        for i in idxs[-num_reserve:]
    ] + [
        {"text": unsup_dataset[i]["text"], "label": unsup_dataset[i]["label"]}
        for i in range(8000)
    ]
    train_dataset = [
        {"text": train_dataset[i]["text"], "label": train_dataset[i]["label"]}
        for i in idxs[:-num_reserve]
    ]
    test_dataset = datasets["test"]
    datasets = {"train": train_dataset, "validation": dev_dataset, "test": test_dataset}
    return datasets


def load_amazon_reviews_multi(language):
    dataset = load_dataset(
        "amazon_reviews_multi", language
    )  # all_languages de fr en es ja zh
    # product_category
    labels = sorted(list(set(dataset["train"]["product_category"])))[::-1]
    print("Numbers of labels Amazon", len(labels))
    dict_labels = {label: i for i, label in enumerate(labels)}
    datasets = {
        "train": dataset["train"],
        "validation": dataset["train"],
        "test": dataset["test"],
    }
    new_datasets = {"train": [], "validation": [], "test": []}
    for split_name in new_datasets.keys():
        arr = datasets[split_name]["product_category"]
        for k, v in tqdm(dict_labels.items()):
            arr = [x.replace(k, str(v)) for x in arr]
        arr = [int(i) for i in arr]
        review_body = datasets[split_name]["review_body"]
        for i in tqdm(range(len(arr))):
            new_datasets[split_name].append(
                {
                    "label": arr[i],
                    "review_body": review_body[i],
                }
            )
    return new_datasets


def load_tweet_eval():
    datasets = load_dataset("tweet_eval", "emoji")  # /plain_text')
    datasets = {
        "train": datasets["train"],
        "validation": datasets["train"],
        "test": datasets["test"],
    }
    return datasets


def load_go_emotions():
    datasets = load_dataset("go_emotions", "simplified")  # label = json
    datasets = {
        "train": datasets["train"],
        "validation": datasets["train"],
        "test": datasets["test"],
    }
    new_datasets = {"train": [], "validation": [], "test": []}
    for split_name in new_datasets.keys():
        arr = datasets[split_name]["labels"]
        arr = [x[0] for x in arr]
        texts = datasets[split_name]["text"]
        for i in tqdm(range(len(arr))):
            new_datasets[split_name].append(
                {
                    "label": arr[i],
                    "text": texts[i],
                }
            )

    new_datasets["validation"] += new_datasets["train"][:5000]
    return new_datasets


def load_sst2():
    datasets = load_dataset("glue", "sst2")

    train = [{"text": x["sentence"], "label": x["label"]} for x in datasets["train"]]
    validation = [
        {"text": x["sentence"], "label": x["label"]} for x in datasets["validation"]
    ]
    test = [{"text": x["sentence"], "label": x["label"]} for x in datasets["test"]]

    datasets = {"train": train, "validation": validation, "test": test}
    return datasets


def prep_model(model_name, config: Optional[Dict] = None):
    if config is None:
        config = {"label": 2}

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=config["label"]
    )

    return model, tokenizer
