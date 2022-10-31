from configs.datasets_configs import DATASETS_CONFIGS
from toddbenchmark.generation_data import prep_dataset, prep_model
from datasets import load_dataset


if __name__ == "__main__":
    model, tokenizer = prep_model("t5-small")

    print("Available datasets configs:")
    for k, v in DATASETS_CONFIGS.items():
        try:
            print(k)
            print(v)
            _, _, _ = prep_dataset(
                v["dataset_name"],
                v["dataset_config"],
                tokenizer,
            )
        except:
            print(f"Failed to load dataset {k}")
            print(v)
            pass
