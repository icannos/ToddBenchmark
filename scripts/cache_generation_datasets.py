from toddbenchmark.generation_datasets_configs import DATASETS_CONFIGS
from toddbenchmark.generation_datasets import prep_dataset, prep_model

import traceback

if __name__ == "__main__":
    model, tokenizer = prep_model("t5-small")

    print("Available datasets configs:")
    for k, v in DATASETS_CONFIGS.items():
        try:
            _, _, _ = prep_dataset(
                v["dataset_name"],
                v["dataset_config"],
                tokenizer,
                train_max_size=-1,
                validation_max_size=-1,
                test_max_size=-1,
            )

            print(k)
            print(v)
        except Exception as e:
            print(f"\033[91mFailed to load dataset {k}")
            print(v)
            print("\033[0m")
            print(e)
            print(traceback.format_exc())
            pass
