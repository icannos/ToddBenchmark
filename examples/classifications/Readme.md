# Classification OOD examples


## Finetuning a model on a classification dataset

```bash
python finetune_classification.py --dataset_config sst2 --model_name_or_path distilbert-base-uncased 
```

## Using the finetuned model to detect OOD

```bash
python evaluate_classification.py --model_name output_finetuning/distilbert-base-uncased-sst2/checkpoint-1000/ \
                                  --in_config sst2 \
                                  --out_configs imdb mnli \
                                  --validation_size 1000 \
                                  --test_size 5000
```