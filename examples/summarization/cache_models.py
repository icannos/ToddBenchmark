from toddbenchmark.generation_datasets import prep_model

models = [
    "google/pegasus-xsum",
    "sshleifer/distilbart-cnn-12-6",
    "google/pegasus-cnn_dailymail",
    "philschmid/bart-large-cnn-samsum",
]

for model_name in models:
    model = prep_model(model_name)

