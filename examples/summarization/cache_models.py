from toddbenchmark.generation_datasets import prep_model

models = [
    "google/pegasus-xsum",
    "sshleifer/distilbart-cnn-12-6",
    "google/pegasus-cnn_dailymail",
    "philschmid/bart-large-cnn-samsum",
    "google/flan-t5-base",
    "google/flan-t5-large",
]

for model_name in models:
    model = prep_model(model_name)
