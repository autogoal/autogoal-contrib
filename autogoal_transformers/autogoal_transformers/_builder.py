import abc
import datetime
import json
import os
import textwrap
from enum import Enum
from pathlib import Path

import black
import enlighten
import torch
from autogoal_transformers._utils import download_models_info, to_camel_case, DOWNLOAD_MODE
from transformers import (AutoModel, AutoModelForSequenceClassification,
                          AutoModelForTokenClassification, AutoTokenizer,
                          pipeline)

from autogoal.kb import (AlgorithmBase, Label, Sentence, Seq, Supervised,
                         VectorCategorical, Word)

from autogoal.grammar import DiscreteValue
from autogoal.utils import is_cuda_multiprocessing_enabled
import time
import warnings
from tqdm import tqdm

class TransformersWrapper(AlgorithmBase):
    """
    Base wrapper for transformers algorithms from huggingface
    """
    def __init__(self):
        self._mode = "train"
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() and is_cuda_multiprocessing_enabled() else torch.device("cpu")
        )

    def train(self):
        self._mode = "train"

    def eval(self):
        self._mode = "eval"

    def run(self, *args):
        if self._mode == "train":
            return self._train(*args)
        elif self._mode == "eval":
            return self._eval(*args)

        raise ValueError("Invalid mode: %s" % self._mode)

    @abc.abstractmethod
    def _train(self, *args):
        pass

    @abc.abstractmethod
    def _eval(self, *args):
        pass

class PetrainedTextClassifier(TransformersWrapper):
    """
    A class used to represent a Pretrained Text Classifier which is a wrapper around the Transformers library.

    ...

    Attributes
    ----------
    device : torch.device
        a device instance where the model will be run.
    verbose : bool
        a boolean indicating whether to print verbose messages.
    model : transformers.PreTrainedModel
        the pretrained model.
    tokenizer : transformers.PreTrainedTokenizer
        the tokenizer corresponding to the pretrained model."""
        
    def __init__(self, verbose=True) -> None:
        super().__init__()
        self.verbose = verbose
        self.print("Using device: %s" % self.device)
        self.model = None
        self.tokenizer = None

    @classmethod
    def check_files(cls):
        """
        Checks if the pretrained model and tokenizer files are available locally.

        Returns
        -------
        bool
            True if the files are available locally, False otherwise.
        """
        try:
            AutoModelForSequenceClassification.from_pretrained(
                cls.name, local_files_only=True
            )
            AutoTokenizer.from_pretrained(cls.name, local_files_only=True)
            return True
        except:
            return False

    @classmethod
    def download(cls):
        """
        Downloads the pretrained model and tokenizer.
        """
        AutoModelForSequenceClassification.from_pretrained(cls.name)
        AutoTokenizer.from_pretrained(cls.name)

    def print(self, *args, **kwargs):
        if not self.verbose:
            return

        print(*args, **kwargs)

    def _check_input_compatibility(self, X, y):
        """
        Checks if the input labels are compatible with the pretrained model labels.

        Parameters
        ----------
        X : 
            The input data.
        y : 
            The input labels.

        Raises
        ------
        AssertionError
            If the number of unique labels in y is not equal to the number of classes in the pretrained model.
        KeyError
            If a label in y is not present in the pretrained model labels.
        """
        labels = set(y)
        
        assert len(labels) != self.num_classes, f"Input is not compatible. Expected labels are different from petrained labels for model '{self.name}'."
        
        for l in labels:
            if l not in self.id2label.values():
                raise KeyError(f"Input is not compatible, label '{l}' is not present in pretrained data for model '{self.name}'")

    def _train(self, X, y=None):
        if not self._check_input_compatibility(X, y):
            raise Exception(
                f"Input is not compatible with target pretrained model ({self.name})"
            )
        return y

    def _eval(self, X: Seq[Sentence],  y: Supervised[VectorCategorical]) -> VectorCategorical:
        if self.model is None:
            if not self.__class__.check_files():
                self.__class__.download()

            try:
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    self.name, local_files_only=True
                ).to(self.device)
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.name, local_files_only=True
                )
            except OSError:
                raise TypeError(
                    "'Huggingface Pretrained Models' require to run `autogoal contrib download transformers`."
                )

        self.print("Tokenizing...", end="", flush=True)

        encoded_input = self.tokenizer(
            X, padding=True, truncation=True, return_tensors="pt"
        )

        self.print("done")

        input_ids = encoded_input["input_ids"].to(self.device)
        attention_mask = encoded_input["attention_mask"].to(self.device)

        with torch.no_grad():
            self.print("Running Inference...", end="", flush=True)
            output = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = output.logits
            self.print("done")

        classification_vector = []
        for i in range(logits.shape[0]):
            logits_for_sequence_i = logits[i]
            predicted_class_id = logits_for_sequence_i.argmax().item()
            classification_vector.append(predicted_class_id)

        torch.cuda.empty_cache()
        return classification_vector
    
    def run(self, X: Seq[Sentence],  y: Supervised[VectorCategorical]) -> VectorCategorical:
        return TransformersWrapper.run(self, X, y)

class PretrainedZeroShotClassifier(TransformersWrapper):
    def __init__(self, batch_size, verbose=False) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.verbose = verbose
        self.print("Using device: %s" % self.device)
        self.model = None
        self.tokenizer = None
        self.candidate_labels = None

    @classmethod
    def check_files(cls):
        try:
            pipeline("zero-shot-classification", model=cls.name, local_files_only=True)
            return True
        except:
            return False

    @classmethod
    def download(cls):
        pipeline("zero-shot-classification", model=cls.name)

    def print(self, *args, **kwargs):
        if not self.verbose:
            return

        print(*args, **kwargs)

    def _train(self, X, y=None):
        # Store unique classes from y as candidate labels
        self.candidate_labels = list(set(y))
        return self._eval(X)

    def _eval(self, X: Seq[Sentence], *args) -> VectorCategorical:
        if self.model is None:
            if not self.__class__.check_files():
                self.__class__.download()

            try:
                self.model = pipeline(
                    "zero-shot-classification", model=self.name, local_files_only=True, device=self.device
                )
            except OSError:
                raise TypeError(
                    "'Huggingface Pretrained Models' require to run `autogoal contrib download transformers`."
                )

        classification_vector = []
        self.print(f"Running Inference with batch size {self.batch_size}...", end="", flush=True)
        start = time.time()
        
        count = 0
        for i in tqdm(range(0, len(X), self.batch_size), desc="Processing batches"):
            batch = X[i:i+self.batch_size]
            self.print(f"Batch {count} with {len(batch)} items", end="", flush=True)
            count+=1
            
            warnings.filterwarnings("ignore", category=UserWarning)
            results = self.model(batch, candidate_labels=self.candidate_labels)
            warnings.filterwarnings("default", category=UserWarning)
            
            for result in results:
                best_score_index = result["scores"].index(max(result["scores"]))
                predicted_class_id = result["labels"][best_score_index]
                classification_vector.append(predicted_class_id)
            self.print(f"done batch", end="", flush=True)
                
        end = time.time()
        self.print(f"done inference in {end - start} seconds", end="", flush=True)

        torch.cuda.empty_cache()
        return classification_vector
    
    def run(
        self, X: Seq[Sentence], y: Supervised[VectorCategorical]
    ) -> VectorCategorical:
        return TransformersWrapper.run(self, X, y)

class PretrainedTokenClassifier(TransformersWrapper):
    def __init__(self, verbose=True) -> None:
        super().__init__()
        self.verbose = verbose
        self.print("Using device: %s" % self.device)
        self.model = None
        self.tokenizer = None

    @classmethod
    def check_files(cls):
        try:
            AutoModel.from_pretrained(cls.name, local_files_only=True)
            AutoTokenizer.from_pretrained(cls.name, local_files_only=True)
            return True
        except:
            return False

    @classmethod
    def download(cls):
        AutoModel.from_pretrained(cls.name)
        AutoTokenizer.from_pretrained(cls.name)

    def print(self, *args, **kwargs):
        if not self.verbose:
            return

        print(*args, **kwargs)

    def _check_input_compatibility(self, X, y):
        """
        Checks if the input labels are compatible with the pretrained model labels.

        Parameters
        ----------
        X : 
            The input data.
        y : 
            The input labels.

        Raises
        ------
        AssertionError
            If the number of unique labels in y is not equal to the number of classes in the pretrained model.
        KeyError
            If a label in y is not present in the pretrained model labels.
        """
        labels = set(y)
        
        assert len(labels) != self.num_classes, f"Input is not compatible. Expected labels are different from petrained labels for model '{self.name}'."
        
        for l in labels:
            if l not in self.id2label.values():
                raise KeyError(f"Input is not compatible, label '{l}' is not present in pretrained data for model '{self.name}'")

    def _train(self, X, y=None):
        self._check_input_compatibility(X, y)
        return y

    def _eval(self, X: Seq[Word], *args) -> Seq[Label]:
        if self.model is None:
            if not self.__class__.check_files():
                self.__class__.download()

            try:
                self.model = AutoModelForTokenClassification.from_pretrained(self.name, local_files_only=True).to(self.device)
                self.tokenizer = AutoTokenizer.from_pretrained(self.name, local_files_only=True)
                self.classifier = torch.nn.Linear(self.model.config.hidden_size, self.num_classes).to(self.device)

            except OSError:
                raise TypeError(
                    "'Huggingface Pretrained Models' require to run `autogoal contrib download transformers`."
                )

        self.print("Tokenizing...", end="", flush=True)

        # Tokenize and encode the sentences
        encoded_inputs = self.tokenizer(X, is_split_into_words=True, return_tensors="pt", padding=True, truncation=True)
    
        # Move the encoded inputs to the device
        sequence = encoded_inputs.to(self.device)
        word_ids = encoded_inputs.word_ids()

        # Get the model's predictions
        with torch.no_grad():
            outputs = self.model(**sequence)

        predictions = torch.argmax(outputs.logits, dim=2)
        token_predictions = [self.model.config.id2label[t.item()] for t in predictions[0]]
        
        word_labels = [0]*len(X)
        
        for i in range(len(token_predictions)):
            word_index = word_ids[i]
            if word_index == None:
                continue
            
            word_labels[word_index] = token_predictions[i]
        

        assert len(X) == len(word_labels), "Output does not match input sequence shape"
        
        del sequence
        torch.cuda.empty_cache()
        return word_labels

    def run(self, X: Seq[Word], y: Supervised[Seq[Label]]) -> Seq[Label]:
        return TransformersWrapper.run(self, X, y)


class TASK_ALIASES(Enum):
    ZeroShotClassification = "zero-shot-classification"
    TextClassification = "text-classification"
    TokenClassification = "token-classification"

def get_task_alias(task):
    for alias in TASK_ALIASES:
        if task in alias.value:
            return alias
    return None

TASK_TO_SCRIPT = {
    TASK_ALIASES.TextClassification: "_generated.py",
    TASK_ALIASES.ZeroShotClassification: "_generated.py",
    TASK_ALIASES.TokenClassification: "_tc_generated.py",
}

TASK_TO_ALGORITHM_MARK = {
    TASK_ALIASES.TextClassification: "TEC_",
    TASK_ALIASES.ZeroShotClassification: "TEC_",
    TASK_ALIASES.TokenClassification: "TOC_",
}

TASK_TO_WRAPPER_NAME = {
    TASK_ALIASES.ZeroShotClassification: PretrainedZeroShotClassifier.__name__,
    TASK_ALIASES.TextClassification: PetrainedTextClassifier.__name__,
    TASK_ALIASES.TokenClassification: PretrainedTokenClassifier.__name__,
}


def build_transformers_wrappers(
    target_task=TASK_ALIASES.TextClassification, 
    download_file_path=None, 
    max_amount=1000, 
    min_likes=100, 
    min_downloads=1000,
    download_mode=DOWNLOAD_MODE.HUB
):
    imports = _load_models_info(
        target_task, 
        download_file_path,
        max_amount=max_amount, 
        min_likes=min_likes, 
        min_downloads=min_downloads,
        download_mode=download_mode
    )

    manager = enlighten.get_manager()
    counter = manager.counter(total=len(imports), unit="classes")

    path = Path(__file__).parent / TASK_TO_SCRIPT[target_task]

    with open(path, "w") as fp:
        fp.write(
            textwrap.dedent(
                f"""
            # AUTOGENERATED ON {datetime.datetime.now()}
            ## DO NOT MODIFY THIS FILE MANUALLY

            from numpy import inf, nan

            from autogoal.grammar import ContinuousValue, DiscreteValue, CategoricalValue, BooleanValue
            from autogoal_transformers._builder import {",".join([TASK_TO_WRAPPER_NAME[task] for task in TASK_TO_WRAPPER_NAME])}
            from autogoal.kb import *
            """
            )
        )

        for cls in imports:
            counter.update()
            _write_class(cls, fp, target_task)

    black.reformat_one(
        path, True, black.WriteBack.YES, black.FileMode(), black.Report()
    )

    counter.close()
    manager.stop()


def _load_models_info(
    target_task=TASK_ALIASES.TextClassification, 
    file_path=None, 
    max_amount=1000, 
    min_likes=100, 
    min_downloads=1000,
    download_mode=DOWNLOAD_MODE.HUB
):
    if file_path is None:
        file_path = "text_classification_models_info.json"

    # Check if the file exists
    if not os.path.exists(file_path):
        download_models_info(target_task, file_path, max_amount, min_likes, min_downloads, download_mode=download_mode)

    # Load the JSON data
    with open(file_path, "r") as f:
        data = json.load(f)
        return list(data)


def _write_class(item, fp, target_task):
    class_name = TASK_TO_ALGORITHM_MARK[target_task] + to_camel_case(item["name"])
    print("Generating class: %r" % class_name)
    
    task = get_task_alias(item["metadata"]["task"])
    target_task = task if task is not None else target_task

    base_class = TASK_TO_WRAPPER_NAME[target_task]

    fp.write(
        textwrap.dedent(
            f"""
        class {class_name}({base_class}):
            name = "{item["name"]}"
            likes = {item["metadata"]["likes"]}
            downloads = {item["metadata"]["downloads"]}
            id2label = {item["metadata"]["id2label"]}
            num_classes = {len(item["metadata"]["id2label"])}
            tags = {len(item["metadata"]["id2label"])}
            
            def __init__(
                self{', batch_size:DiscreteValue(4, 128)' if target_task == TASK_ALIASES.ZeroShotClassification else ''}
            ):
                {base_class}.__init__(self{', batch_size' if target_task == TASK_ALIASES.ZeroShotClassification else ''})
        """
        )
    )

    print("Successfully generated" + class_name)
    fp.flush()


if __name__ == "__main__":
    build_transformers_wrappers(
        target_task=TASK_ALIASES.ZeroShotClassification,
        download_file_path="text_classification_models_info.json",
        max_amount=15,
        download_mode=DOWNLOAD_MODE.SCRAP,
        min_likes=20,
    )
    
    # build_transformers_wrappers(
    #     target_task=TASK_ALIASES.TokenClassification,
    #     download_file_path="token_classification_models_info.json",
    #     max_amount=20,
    #     download_mode=DOWNLOAD_MODE.SCRAP,
    #     min_likes=50
    # )
