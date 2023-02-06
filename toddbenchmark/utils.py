import json
from json import encoder

encoder.FLOAT_REPR = lambda o: format(o, ".3f")


def dump_json(records, path, append=False):

    if append:
        with open(path, "r") as f:
            existing_records = json.load(f)
        records = existing_records | records

    with open(path, "w") as f:
        json.dump(records, f)


def sanitize_model_name(model_name: str) -> str:
    """
    Sanitize the model name to be used as a file name
    :param model_name: Model name
    :return: Sanitized model name
    """
    return model_name.replace("/", "_")

def mk_file_name(model_name: str, dataset_in_config, dataset_out_config) -> str:
    """
    Make a file name for the results
    :param model_name: Model name
    :param dataset_in_config: Dataset in config
    :param dataset_out_config: Dataset out config
    :return: File name
    """
    model_name = sanitize_model_name(model_name)
    return f"{model_name}_{dataset_in_config}_{dataset_out_config}.json"
