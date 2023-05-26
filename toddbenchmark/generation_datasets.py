import random
from typing import Tuple

from datasets import load_dataset, Dataset, load_from_disk
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
import torch

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
    dataset = load_dataset(
        dataset_name,
        dataset_config,
    )

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
    dataset = load_dataset(
        "wiki_movies",
    )

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
    dataset = load_dataset(
        dataset_name,
        dataset_config,
    )

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
    datasets = try_load_dataset_config(
        dataset_name,
        dataset_config,
    )
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
            dataset_name,
            lang1=lang1,
            lang2=lang2,
        )
    except ValueError:
        try:
            dataset = load_dataset(
                dataset_name,
                lang1=lang2,
                lang2=lang1,
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
    dataset = load_dataset(
        dataset_name,
        dataset_config,
    )

    train = [(d["review_title"], d["review_title"]) for d in dataset["train"]]
    val = [(d["review_title"], d["review_title"]) for d in dataset["validation"]]
    test = [(d["review_title"], d["review_title"]) for d in dataset["test"]]

    return {"train": train, "validation": val, "test": test}


## Summarization datasets


def load_xsum_dataset(dataset_name, dataset_config):
    dataset = load_dataset(
        dataset_name,
    )

    train = [(d["document"], d["summary"]) for d in dataset["train"]]
    val = [(d["document"], d["summary"]) for d in dataset["validation"]]
    test = [(d["document"], d["summary"]) for d in dataset["test"]]

    return {"train": train, "validation": val, "test": test}


def load_cnndm_dataset(dataset_name, dataset_config):
    dataset = load_dataset(
        dataset_name,
        dataset_config,
    )

    train = [(d["article"], d["highlights"]) for d in dataset["train"]]
    val = [(d["article"], d["highlights"]) for d in dataset["validation"]]
    test = [(d["article"], d["highlights"]) for d in dataset["test"]]

    return {"train": train, "validation": val, "test": test}


def load_billsum_dataset(dataset_name, dataset_config):
    dataset = load_dataset(
        dataset_name,
    )

    train = [(d["text"], d["summary"]) for d in dataset["train"]]
    train, val = train[:10000], train[10000:]
    test = [(d["text"], d["summary"]) for d in dataset["test"]]

    return {"train": train, "validation": val, "test": test}


def load_multi_news_dataset(dataset_name, dataset_config):
    dataset = load_dataset(
        dataset_name,
    )

    train = [(d["document"], d["summary"]) for d in dataset["train"]]
    val = [(d["document"], d["summary"]) for d in dataset["validation"]]
    test = [(d["document"], d["summary"]) for d in dataset["test"]]

    return {"train": train, "validation": val, "test": test}


## Question Answering Datasets


def load_web_questions_dataset(dataset_name, dataset_config):
    dataset = load_dataset("web_questions")

    train = [(d["question"], d["answers"][0]) for d in dataset["train"]]

    val = train[:1000]
    test = [(d["question"], d["answers"][0]) for d in dataset["test"]]

    return {"train": train, "validation": val, "test": test}


def load_openbookqa_dataset(dataset_name, dataset_config):
    # dataset config: answerable, unanswerable

    dataset = load_dataset("openbookqa", "additional")

    def mk_input(x):
        txt = f"Context:{x['fact1']} ; Question: {x['question_stem']} ; Choices: {' - '.join(x['choices']['text'])}"
        return txt

    def mk_target(x):
        idx_answer = x["choices"]["label"].index(x["answerKey"])
        return x["choices"]["text"][idx_answer]

    all_possible_answers = {}
    for split in dataset:
        all_possible_answers[split] = []
        for x in dataset[split]:
            all_possible_answers[split].extend(x["choices"]["text"])

    def remplace_answer_by_random(x):
        answer_key = x["answerKey"]
        idx_answer = x["choices"]["label"].index(answer_key)
        good_answer = x["choices"]["text"][idx_answer]

        random_answer = random.choice(all_possible_answers[split])
        while random_answer == good_answer:
            random_answer = random.choice(all_possible_answers[split])
        x["choices"]["text"][idx_answer] = random_answer

        return x

    if dataset_config == "answerable":
        dataset = {
            split: [(mk_input(x), mk_target(x)) for x in dataset[split]]
            for split in dataset
        }
    elif dataset_config == "unanswerable":
        altered_dataset = {
            split: [remplace_answer_by_random(x) for x in dataset[split]]
            for split in dataset
        }
        dataset = {
            split: [(mk_input(x), "None") for x in altered_dataset[split]]
            for split in altered_dataset
        }
    else:
        raise ValueError(
            f"Invalid dataset config: {dataset_config}, should be answerable or unanswerable"
        )

    return dataset


def load_ai2arc(dataset_name, dataset_config):
    answerable, config = dataset_config.split("_")

    dataset = load_dataset(
        "ai2_arc",
        config,
    )

    def mk_input(x):
        txt = f"Context: ; Question: {x['question']} ; Choices: {' - '.join(x['choices']['text'])}"
        return txt

    def mk_target(x):
        answer_key = x["answerKey"]
        idx_answer = x["choices"]["label"].index(answer_key)
        return x["choices"]["text"][idx_answer]

    all_possible_answers = {}
    for split in dataset:
        all_possible_answers[split] = []
        for x in dataset[split]:
            all_possible_answers[split].extend(x["choices"]["text"])

    def remplace_answer_by_random(x):
        answer_key = x["answerKey"]
        idx_answer = x["choices"]["label"].index(answer_key)
        good_answer = x["choices"]["text"][idx_answer]

        random_answer = random.choice(all_possible_answers[split])
        while random_answer == good_answer:
            random_answer = random.choice(all_possible_answers[split])
        x["choices"]["text"][idx_answer] = random_answer

        return x

    if answerable == "answerable":
        dataset = {
            split: [(mk_input(x), mk_target(x)) for x in dataset[split]]
            for split in dataset
        }
    elif answerable == "unanswerable":
        altered_dataset = {
            split: [remplace_answer_by_random(x) for x in dataset[split]]
            for split in dataset
        }
        dataset = {
            split: [(mk_input(x), "None") for x in altered_dataset[split]]
            for split in altered_dataset
        }

    else:
        raise ValueError(
            f"Invalid dataset config: {dataset_config}, should be answerable or unanswerable"
        )

    return dataset


def load_sciq(dataset_name, dataset_config):
    # dataset config: answerable, unanswerable)

    dataset = load_dataset("sciq", "additional")

    def mk_input(x):
        choices = [
            x["distractor1"],
            x["distractor2"],
            x["distractor3"],
            x["correct_answer"],
        ]
        # shuffle
        random.shuffle(choices)

        txt = f"Context:{x['support']} ; Question: {x['question']} ; Choices: {' - '.join(choices)}"
        return txt

    def mk_target(x):
        return x["correct_answer"]

    all_possible_answers = {}
    for split in dataset:
        all_possible_answers[split] = []
        for x in dataset[split]:
            all_possible_answers[split].extend(
                [
                    x["distractor1"],
                    x["distractor2"],
                    x["distractor3"],
                    x["correct_answer"],
                ]
            )

    def remplace_answer_by_random(x):
        good_answer = x["correct_answer"]
        random_answer = random.choice(all_possible_answers[split])
        while random_answer == good_answer:
            random_answer = random.choice(all_possible_answers[split])
        x["correct_answer"] = random_answer

        return x

    if dataset_config == "answerable":
        dataset = {
            split: [(mk_input(x), mk_target(x)) for x in dataset[split]]
            for split in dataset
        }
    elif dataset_config == "unanswerable":
        altered_dataset = {
            split: [remplace_answer_by_random(x) for x in dataset[split]]
            for split in dataset
        }
        dataset = {
            split: [(mk_input(x), "None") for x in altered_dataset[split]]
            for split in altered_dataset
        }
    else:
        raise ValueError(
            f"Invalid dataset config: {dataset_config}, should be answerable or unanswerable"
        )

    return dataset


def load_trivia_qa(dataset_name, dataset_config="closed_book"):

    if dataset_config == "closed_book":
        dataset = load_dataset("trivia_qa", "rc.nocontext")
        train = [
            (sample["question"], sample["answer"]["value"])
            for sample in dataset["train"]
        ]
        validation = [
            (sample["question"], sample["answer"]["value"])
            for sample in dataset["validation"]
        ]
        test = [
            (sample["question"], sample["answer"]["value"])
            for sample in dataset["test"]
        ]

        return {"train": train, "validation": validation, "test": test}

    elif dataset_config == "open_book_answerable":
        dataset = load_dataset("trivia_qa", "rc")

        def mk_input(x):
            context = x["search_results"]["search_context"][0].replace("\n", " ")
            txt = f"Context: {context} ; Question: {x['question']}"
            return txt

        def mk_target(x):
            return x["answer"]["value"]

        train = [(mk_input(sample), mk_target(sample)) for sample in dataset["train"]]
        validation = [
            (mk_input(sample), mk_target(sample)) for sample in dataset["validation"]
        ]
        test = [(mk_input(sample), mk_target(sample)) for sample in dataset["test"]]

        return {"train": train, "validation": validation, "test": test}

        # elif dataset_config == "open_book_unanswerable":
        #
        #     all_questions = []
        #     for split in dataset:
        #         all_questions.extend([sample["question"] for sample in dataset[split]])
        #
        #     def mk_input(x):
        #         # select a random question
        #         random_question = random.choice(all_questions)
        #         context = x["search_results"]["search_context"][0].replace("\n", " ")
        #         txt = f"Context: {context} ; Question: {random_question}"
        #         return txt

        # def mk_target(x):
        #     return "None"

        # train = [(mk_input(sample), mk_target(sample)) for sample in dataset["train"]]
        # validation = [
        #     (mk_input(sample), mk_target(sample)) for sample in dataset["validation"]
        # ]
        # test = [(mk_input(sample), mk_target(sample)) for sample in dataset["test"]]
        #
        # return {"train": train, "validation": validation, "test": test}

    else:
        raise ValueError(
            f"Invalid dataset config: {dataset_config}, should be open_book_answerable, open_book_unanswerable or closed_book"
        )


def load_coqa(dataset_name, dataset_config="answerable"):
    dataset = load_dataset("coqa")

    _dataset = {}
    for split in dataset:
        _dataset[split] = []
        for sample in dataset[split]:
            for question, answer in zip(
                sample["questions"], sample["answers"]["input_text"]
            ):
                _dataset[split].append((sample["story"], question, answer))

    def mk_input(x):
        return f"Context: {x[0]} ; Question: {x[1]}"

    def mk_target(x):
        return x[2]

    if dataset_config == "answerable":
        dataset = {
            split: [(mk_input(x), mk_target(x)) for x in _dataset[split]]
            for split in _dataset
        }

        return {
            "train": dataset["train"][:3000],
            "validation": dataset["train"][3000:],
            "test": dataset["validation"],
        }

    return dataset


def load_tweetqa(dataset_name, dataset_config):
    dataset = load_dataset(
        "tweet_qa",
    )

    def mk_input(x):
        txt = f"Context:{x['Tweet']} ; Question: {x['Question']}"
        return txt

    def mk_target(x):
        return x["Answer"][0]

    all_possible_answers = {}
    for split in dataset:
        all_possible_answers[split] = []
        for x in dataset[split]:
            all_possible_answers[split].extend(x["Answer"])

    def remplace_answer_by_random(x):
        good_answer = x["Answer"][0]
        random_answer = random.choice(all_possible_answers[split])
        while random_answer == good_answer:
            random_answer = random.choice(all_possible_answers[split])
        x["Answer"][0] = random_answer

        return x

    if dataset_config == "answerable":
        dataset = {
            split: [(mk_input(x), mk_target(x)) for x in dataset[split]]
            for split in dataset
        }
    elif dataset_config == "unanswerable":
        altered_dataset = {
            split: [remplace_answer_by_random(x) for x in dataset[split]]
            for split in dataset
        }
        dataset = {
            split: [(mk_input(x), "None") for x in altered_dataset[split]]
            for split in altered_dataset
        }

    else:
        raise ValueError(
            f"Invalid dataset config: {dataset_config}, should be answerable or unanswerable"
        )

    return dataset


def load_hendrycks(dataset_name, dataset_config):
    topics = [
        "abstract_algebra",
        "anatomy",
        "astronomy",
        "business_ethics",
        "clinical_knowledge",
        "college_biology",
        "college_chemistry",
        "college_computer_science",
        "college_mathematics",
        "college_medicine",
        "college_physics",
        "computer_security",
        "conceptual_physics",
        "econometrics",
        "electrical_engineering",
        "elementary_mathematics",
        "formal_logic",
        "global_facts",
        "high_school_biology",
        "high_school_chemistry",
        "high_school_computer_science",
        "high_school_european_history",
        "high_school_geography",
        "high_school_government_and_politics",
        "high_school_macroeconomics",
        "high_school_mathematics",
        "high_school_microeconomics",
        "high_school_physics",
        "high_school_psychology",
        "high_school_statistics",
        "high_school_us_history",
        "high_school_world_history",
        "human_aging",
        "human_sexuality",
        "international_law",
        "jurisprudence",
        "logical_fallacies",
        "machine_learning",
        "management",
        "marketing",
        "medical_genetics",
        "miscellaneous",
        "moral_disputes",
        "moral_scenarios",
        "nutrition",
        "philosophy",
        "prehistory",
        "professional_accounting",
        "professional_law",
        "professional_medicine",
        "professional_psychology",
        "public_relations",
        "security_studies",
        "sociology",
        "us_foreign_policy",
        "virology",
        "world_religions",
    ]

    train = []
    validation = []
    test = []

    for topic in topics:
        dataset = load_dataset("hendrycks_test", topic)

        def mk_input(x):
            return f"{x['context']} ; Question: {x['question']}"


def load_natural_instructions(dataset_name, dataset_config):
    dataset = load_from_disk(
        "/gpfsdswork/projects/rech/ehz/uwf24rf/ToddProject/datasets/natural-instructions/",
        # verification_mode="no_checks",
    )

    def mk_input(x):
        return f"{x['definition']}.\n {x['inputs']}"

    train = [(mk_input(x), x["targets"]) for x in dataset["train"]]
    validation = [(mk_input(x), x["targets"]) for x in dataset["validation"]]
    test = [(mk_input(x), x["targets"]) for x in dataset["test"]]

    return {"train": train, "validation": validation, "test": test}


def load_quartz(dataset_name, dataset_config):
    dataset = load_dataset(
        "quartz",
    )

    def mk_input(x):
        txt = f"Context:{x['para']} ; Question: {'- '.join(x['choices']['text'])}"
        return txt

    def mk_target(x):
        idx_answer = x["choices"]["label"].index(x["answerKey"])
        return x["choices"]["text"][idx_answer]

    all_possible_answers = {}
    for split in dataset:
        all_possible_answers[split] = []
        for x in dataset[split]:
            all_possible_answers[split].extend(x["choices"]["text"])

    def remplace_answer_by_random(x):
        answer_key = x["answerKey"]
        idx_answer = x["choices"]["label"].index(answer_key)
        good_answer = x["choices"]["text"][idx_answer]

        random_answer = random.choice(all_possible_answers[split])
        while random_answer == good_answer:
            random_answer = random.choice(all_possible_answers[split])
        x["choices"]["text"][idx_answer] = random_answer

        return x

    if dataset_config == "answerable":
        dataset = {
            split: [(mk_input(x), mk_target(x)) for x in dataset[split]]
            for split in dataset
        }

    elif dataset_config == "unanswerable":
        altered_dataset = {
            split: [remplace_answer_by_random(x) for x in dataset[split]]
            for split in dataset
        }
        dataset = {
            split: [(mk_input(x), "None") for x in altered_dataset[split]]
            for split in altered_dataset
        }

    else:
        raise ValueError(
            f"Invalid dataset config: {dataset_config}, should be answerable or unanswerable"
        )

    return dataset


def load_squadv2(dataset_name, dataset_config):
    # dataset config: answerable, unanswerable)

    dataset = load_dataset("squad_v2")

    def mk_input(x):
        txt = f"Context:{x['context']} ; Question: {x['question']}"
        return txt

    def mk_target(x):
        has_answer = len(x["answers"]["answer_start"]) > 0
        return None if not has_answer else x["answers"]["text"][0]

    if dataset_config == "answerable":
        # filter out examples without answer
        dataset = {
            split: [
                (mk_input(x), mk_target(x))
                for x in dataset[split]
                if mk_target(x) is not None
            ]
            for split in dataset
        }
    elif dataset_config == "unanswerable":
        # filter out examples with answer
        dataset = {
            split: [
                (mk_input(x), "None") for x in dataset[split] if mk_target(x) is None
            ]
            for split in dataset
        }
    else:
        raise ValueError(
            f"Invalid dataset config: {dataset_config}, should be answerable or unanswerable"
        )

    train = dataset["train"]
    val = dataset["validation"]

    dataset = {
        "train": train[:-5000],
        "validation": val[:3000] + train[-5000:],
        "test": val[3000:],
    }

    return dataset


def load_race(dataset_name, dataset_config):
    pass


def load_cuad(dataset_name, dataset_config):
    dataset = load_dataset("cuad")

    def mk_input(x):
        txt = f"Context:{x['context']} ; Question: {x['question']}"
        return txt

    def mk_target(x):
        return x["answers"]["text"][0]

    all_possible_answers = {}
    for split in dataset:
        all_possible_answers[split] = []
        for x in dataset[split]:
            all_possible_answers[split].extend(x["answers"]["text"])

    def remplace_answer_by_random(x):
        if len(x["answers"]["text"]) > 0:
            good_answer = x["answers"]["text"][0]
        else:
            good_answer = None

        random_answer = random.choice(all_possible_answers[split])
        while random_answer == good_answer:
            random_answer = random.choice(all_possible_answers[split])

        if good_answer is not None:
            x["answers"]["text"][0] = random_answer
        else:
            x["answers"]["text"].append(random_answer)

        return x

    if dataset_config == "answerable":
        dataset = {
            split: [(mk_input(x), mk_target(x)) for x in dataset[split]]
            for split in dataset
        }
    elif dataset_config == "unanswerable":
        altered_dataset = {
            split: [remplace_answer_by_random(x) for x in dataset[split]]
            for split in dataset
        }
        dataset = {
            split: [(mk_input(x), "None") for x in altered_dataset[split]]
            for split in altered_dataset
        }

    else:
        raise ValueError(
            f"Invalid dataset config: {dataset_config}, should be answerable or unanswerable"
        )

    train = dataset["train"]
    val = dataset["validation"]

    # Correct some sizes
    dataset = {
        "train": train[:-1500],
        "validation": val[:1500] + train[-1500:],
        "test": val[1500:],
    }

    return dataset


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

    elif dataset_name == "web_questions":
        dataset = load_web_questions_dataset(dataset_name, dataset_config)

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

    elif dataset_name == "coqa":
        dataset = load_coqa(dataset_name, dataset_config)

    elif dataset_name == "trivia_qa":
        dataset = load_trivia_qa(dataset_name, dataset_config)

    elif dataset_name == "openbookqa":
        dataset = load_openbookqa_dataset(
            dataset_name,
            dataset_config,
        )
    elif dataset_name == "squad_v2":
        dataset = load_squadv2(
            dataset_name,
            dataset_config,
        )
    elif dataset_name == "cuad":
        dataset = load_cuad(
            dataset_name,
            dataset_config,
        )

    elif dataset_name == "natural_instructions":
        dataset = load_natural_instructions(
            dataset_name,
            dataset_config,
        )

    elif dataset_name == "ai2_arc":
        dataset = load_ai2arc(
            dataset_name,
            dataset_config,
        )

    elif dataset_name == "sciq":
        dataset = load_sciq(
            dataset_name,
            dataset_config,
        )

    elif dataset_name == "tweetqa":
        dataset = load_tweetqa(
            dataset_name,
            dataset_config,
        )
    elif dataset_name == "quartz":
        dataset = load_quartz(
            dataset_name,
            dataset_config,
        )

    # Summarization datasets
    elif dataset_name == "cnn_dailymail":
        dataset = load_cnndm_dataset(dataset_name, dataset_config)
    elif dataset_name == "xsum":
        dataset = load_xsum_dataset(dataset_name, dataset_config)
    elif dataset_name == "billsum":
        dataset = load_billsum_dataset(dataset_name, dataset_config)
    elif dataset_name == "multi_news":
        dataset = load_multi_news_dataset(dataset_name, dataset_config)
    else:
        raise ValueError("Dataset not supported")

    def to_dict(dataset, split):
        ds = dataset[split]
        return [{"source": s, "target": t} for s, t in ds]

    train = Dataset.from_list(to_dict(dataset, "train")[:train_max_size])
    val = Dataset.from_list(to_dict(dataset, "validation")[:validation_max_size])
    test = Dataset.from_list(to_dict(dataset, "test")[:test_max_size])

    return train, val, test


def prep_model(model_name, device="cuda", src_lang=None):
    if model_name == "microsoft/DialoGPT-medium" or model_name == "tosin/dialogpt_mwoz":
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.truncation_side = "left"
        tokenizer.model_max_length = 50
        model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map="auto", load_in_8bit=True, torch_dtype=torch.bfloat16
        )

    if model_name == "nomic-ai/gpt4all-j":
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map="auto", torch_dtype=torch.float16
        )
        added_tokens = tokenizer.add_special_tokens(
            {"bos_token": "<s>", "eos_token": "</s>", "pad_token": "<pad>"}
        )
        if added_tokens > 0:
            model.resize_token_embeddings(len(tokenizer))

    elif "google/flan" in model_name:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
        )
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name, device_map="auto", torch_dtype=torch.bfloat16
        )

    elif "nllb" in model_name:
        tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang=src_lang)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name, device_map="auto", torch_dtype=torch.bfloat16
        )

    elif "bloomz" in model_name:
        tokenizer = AutoTokenizer.from_pretrained("bigscience/bloomz")
        model = AutoModelForCausalLM.from_pretrained(
            "bigscience/bloomz-560m",
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )

    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        model = model.to(device)

    return model, tokenizer


def prep_inputs(x, tokenizer, dataset_name):
    if dataset_name == "daily_dialog" or dataset_name == "movieqa":
        inputs = tokenizer(x, return_tensors="pt", truncation=True)
    else:
        inputs = tokenizer(x, return_tensors="pt", truncation=True)

    return inputs


if __name__ == "__main__":
    pass
