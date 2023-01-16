from bert_score import BERTScorer
from sacrebleu import BLEU
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class BERTScorerWrapper:
    def __new__(cls, *args, **kwargs):
        return BERTScorer(*args, **kwargs)


class BLEUWrapper:
    def __new__(cls, *args, **kwargs):
        return BLEU(*args, **kwargs)


class AutoTokenizerWrapper:
    def __new__(cls, *args, **kwargs):
        return AutoTokenizer.from_pretrained(*args, **kwargs)


class AutoModelForSeq2SeqLMWrapper:
    def __new__(cls, *args, **kwargs):
        return AutoModelForSeq2SeqLM.from_pretrained(*args, **kwargs)
