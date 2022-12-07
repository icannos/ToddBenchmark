from toddbenchmark.classification_datasets_configs import DATASETS_CONFIGS
from toddbenchmark.classification_datasets import prep_dataset, prep_model

if __name__ == "__main__":
    print("Classification datasets:")
    model, tokenizer = prep_model("distilbert-base-uncased")

    print("Available datasets configs:")
    for config_name, config in DATASETS_CONFIGS.items():
        try:
            prep_dataset(config_name, DATASETS_CONFIGS[config_name], tokenizer)

            print(config_name)
            print(config)
        except Exception as e:
            print(f"\033[91mFailed to load dataset {config_name}")
            print(config)
            print("\033[0m")
            print(e)
