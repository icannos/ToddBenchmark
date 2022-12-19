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


# transpose list of list of dict into dict of list
