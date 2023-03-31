from toddbenchmark.generation_datasets_configs import TRANSLATION_DATASETS
from toddbenchmark.generation_datasets import prep_model


# prep_model("facebook/wmt19-de-en")
prep_model("facebook/wmt19-de-en")
print("Available Translation datasets configs:")
for k, v in TRANSLATION_DATASETS.items():
    print(k)
    print(v)
    print("===========")

print("Model names associated with translation datasets:")
small_models = []
for k, v in TRANSLATION_DATASETS.items():
    if "model_config" not in v:
        src, tgt = v["dataset_config"].split("-")
        src, tgt = src[:2], tgt[:2]
    else:
        src, tgt = v["model_config"].split("-")
    small_models.append(f"Helsinki-NLP/opus-mt-{src}-{tgt}")

print(" ".join(small_models))


# Write sbatch script


# model, in-ds pairs:
model_inds = []
for k, v in TRANSLATION_DATASETS.items():
    if "model_config" not in v:
        src, tgt = v["dataset_config"].split("-")
        src, tgt = src[:2], tgt[:2]
    else:
        src, tgt = v["model_config"].split("-")
    model_inds.append((f"Helsinki-NLP/opus-mt-{src}-{tgt}", k))
    try:
        pass
        prep_model(f"Helsinki-NLP/opus-mt-{src}-{tgt}")
        print("OK")
    except Exception as e:
        print("BUG:" + k)
        print(f"Helsinki-NLP/opus-mt-{src}-{tgt}")
        print(e)

print(" ".join([f"{m},{i}" for m, i in model_inds]))
print(" ".join([f"{ds}" for ds in TRANSLATION_DATASETS.keys()]))


# Facebook wmt 19:

# facebook/wmt19-ru-en
#
# facebook/wmt19-ru-en
