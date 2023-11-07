import abc
import json
import datetime
import textwrap
import os
from pathlib import Path
from autogoal.kb import (
    AlgorithmBase,
    Supervised,
    VectorDiscrete,
    Seq,
    Sentence,
)
from autogoal_transformers._utils import (
    download_models_info,
    to_camel_case,
)
import black
import enlighten
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from enum import Enum


class TransformersWrapper(AlgorithmBase):
    def __init__(self):
        self._mode = "train"

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
    def __init__(self, verbose=False) -> None:
        super().__init__()
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.verbose = verbose
        self.print("Using device: %s" % self.device)
        self.model = None
        self.tokenizer = None

    @classmethod
    def check_files(cls):
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
        AutoModelForSequenceClassification.from_pretrained(cls.name)
        AutoTokenizer.from_pretrained(cls.name)

    def print(self, *args, **kwargs):
        if not self.verbose:
            return

        print(*args, **kwargs)

    def _check_input_compatibility(self, X, y):
        return len(set(y)) <= self.num_classes

    def _train(self, X, y=None):
        if not self._check_input_compatibility(X, y):
            raise Exception(
                f"Input is not compatible with target pretrained model ({self.name})"
            )
        return y

    def _eval(self, X: Seq[Sentence], *args) -> VectorDiscrete:
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

        return classification_vector


class PretrainedZeroShotClassifier(TransformersWrapper):
    def __init__(self, verbose=False) -> None:
        super().__init__()
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
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
        return y

    def _eval(self, X: Seq[Sentence], *args) -> VectorDiscrete:
        if self.model is None:
            if not self.__class__.check_files():
                self.__class__.download()

            try:
                self.model = pipeline(
                    "zero-shot-classification", model=self.name, local_files_only=True, device=self.device.index
                )
            except OSError:
                raise TypeError(
                    "'Huggingface Pretrained Models' require to run `autogoal contrib download transformers`."
                )

        self.print("Tokenizing...", end="", flush=True)

        classification_vector = []
        
        for sentence in X:
            result = self.model(sentence, candidate_labels=self.candidate_labels)
            best_score_index = result["scores"].index(max(result["scores"]))
            predicted_class_id = result["labels"][best_score_index]
            classification_vector.append(predicted_class_id)

        return classification_vector


class TASK_ALIASES(Enum):
    ZeroShotClassification = ("zero-shot-classification",)
    TextClassification = ("text-classification",)

def get_task_alias(task):
    for alias in TASK_ALIASES:
        if task in alias.value:
            return alias
    return None

TASK_TO_SCRIPT = {
    TASK_ALIASES.TextClassification: "_generated.py",
}

TASK_TO_WRAPPER_NAME = {
    TASK_ALIASES.ZeroShotClassification: PretrainedZeroShotClassifier.__name__,
    TASK_ALIASES.TextClassification: PetrainedTextClassifier.__name__,
}


def build_transformers_wrappers(
    target_task=TASK_ALIASES.TextClassification, download_file_path=None
):
    imports = _load_models_info(target_task, download_file_path)

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
    target_task=TASK_ALIASES.TextClassification, file_path=None, max_amount=1000
):
    if file_path is None:
        file_path = "text_classification_models_info.json"

    # Check if the file exists
    if not os.path.exists(file_path):
        download_models_info(target_task, file_path, max_amount)

    # Load the JSON data
    with open(file_path, "r") as f:
        data = json.load(f)
        return list(data)


def _write_class(item, fp, target_task):
    class_name = to_camel_case(item["name"])
    print("Generating class: %r" % class_name)
    
    task = get_task_alias(item["metadata"]["task"])
    target_task = task if task is not None else target_task

    input_str = f"X: {repr(Seq[Sentence])}, y: Supervised[VectorCategorical]"
    output_str = "VectorCategorical"
    base_class = TASK_TO_WRAPPER_NAME[target_task]

    fp.write(
        textwrap.dedent(
            f"""
        class {class_name}({base_class}):
            name = "{item["name"]}"
            id2label = {item["metadata"]["id2label"]}
            num_classes = {len(item["metadata"]["id2label"])}
            
            def __init__(
                self
            ):
                {base_class}.__init__(self)

            def run(self, {input_str}) -> {output_str}:
               return {base_class}.run(self, X, y)
        """
        )
    )

    print("Successfully generated" + class_name)
    fp.flush()


if __name__ == "__main__":
    build_transformers_wrappers(
        target_task=TASK_ALIASES.TextClassification,
        download_file_path="text_classification_models_info.json",
    )
