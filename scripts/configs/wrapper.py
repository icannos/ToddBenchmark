from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class AutoTokenizerWrapper:
    def __new__(cls, *args, **kwargs):
        return AutoTokenizer.from_pretrained(*args, **kwargs)


class AutoModelForSeq2SeqLMWrapper:
    def __new__(cls, *args, **kwargs):
        return AutoModelForSeq2SeqLM.from_pretrained(*args, **kwargs)
