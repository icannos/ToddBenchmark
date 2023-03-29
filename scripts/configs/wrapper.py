from typing import Optional
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForMaskedLM, AutoModelForCausalLM


class AutoTokenizerWrapper:
    def __new__(cls, *args, **kwargs):
        return AutoTokenizer.from_pretrained(*args, **kwargs)


class AutoModelForSeq2SeqLMWrapper:
    def __new__(cls, *args, **kwargs):
        return AutoModelForSeq2SeqLM.from_pretrained(*args, **kwargs)


class AutoModelForMaskedLMWrapper:
    def __new__(cls, *args, **kwargs):
        return AutoModelForMaskedLM.from_pretrained(*args, **kwargs)


class AutoModelForCausalLMWrapper:
    def __new__(cls, *args, **kwargs):
        return AutoModelForCausalLM.from_pretrained(*args, **kwargs)


class ExperimentArgs(object):
    def __init__(self,
                 batch_size: int = 8,
                 num_workers: int = 4,
                 max_length: int = 250,
                 num_return_sequences: int = 1,
                 seed: int = 42,
                 output_dir: str = "output/",
                 append: bool = False,
                 validation_size: Optional[int] = None,
                 test_size: Optional[int] = None,
                 device: Optional[str] = None,
                 **kwargs):

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_return_sequences = num_return_sequences
        self.max_length = max_length
        self.seed = seed
        self.kwargs = kwargs
        self.output_dir = output_dir
        self.append = append
        self.test_size = test_size
        self.validation_size = validation_size
        self.device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")

        assert isinstance(self.seed, int)
        assert isinstance(self.num_workers, int)
        assert isinstance(self.batch_size, int)
        assert isinstance(self.num_return_sequences, int)
        assert isinstance(self.max_length, int)
        assert isinstance(self.append, bool)

        assert self.num_workers > 0, "num_workers must be > 0"
        assert self.batch_size > 0, "batch_size must be > 0"
        assert self.num_return_sequences > 0, "num_return_sequences must be > 0"
        assert self.max_length > 0, "max_length must be > 0"
