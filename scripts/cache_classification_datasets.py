from toddbenchmark.classification_datasets_configs import DATASETS_CONFIGS
from toddbenchmark.classification_datasets import load_requested_config, prep_model

if __name__ == "__main__":
    model, tokenizer = prep_model("distilbert-base-uncased")

    print("Available datasets configs:")
    for config_name, config in DATASETS_CONFIGS.items():
        try:
            load_requested_config(config_name, tokenizer)

            print(config_name)
            print(config)
        except Exception as e:
            print(f"\033[91mFailed to load dataset {config_name}")
            print(config)
            print("\033[0m")
            print(e)
            pas
