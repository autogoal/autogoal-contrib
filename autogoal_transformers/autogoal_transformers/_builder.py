import abc
import datetime
import json
import os
import textwrap
from enum import Enum
from pathlib import Path

import black
import enlighten
import numpy as np
import torch
from autogoal_transformers._utils import download_models_info, to_camel_case, DOWNLOAD_MODE
from transformers import (AutoModel, AutoModelForSequenceClassification,
                          AutoModelForTokenClassification, AutoTokenizer,
                          pipeline)

from autogoal.kb import (AlgorithmBase, Label, Sentence, Seq, Supervised,
                         VectorCategorical, Word, MatrixContinuousDense)

from autogoal.grammar import DiscreteValue, CategoricalValue
from autogoal_transformers._utils import TASK_ALIASES
from autogoal.utils._process import is_cuda_multiprocessing_enabled
import time
import warnings
from tqdm import tqdm

class TransformersWrapper(AlgorithmBase):
    """
    Base wrapper for transformers algorithms from huggingface
    """
    @classmethod
    def is_upscalable(cls) -> bool:
        return False
    
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

    def print(self, *args, **kwargs):
        if not self.verbose:
            return

        print(*args, **kwargs)
        
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
            AutoModel.from_pretrained(
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
        AutoModel.from_pretrained(cls.name)
        AutoTokenizer.from_pretrained(cls.name)
    
    def init_model(self):
        if self.model is None:
            if not self.__class__.check_files():
                self.__class__.download()
            try:
                self.model = AutoModel.from_pretrained(
                    self.name, local_files_only=True
                ).to(self.device)
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.name, local_files_only=True
                )
            except OSError as e:
                raise TypeError(
                    f"{self.name} requires to run `autogoal contrib download transformers`."
                )
            except Exception as e:
                raise e
    

class PretrainedWordEmbedding(TransformersWrapper):
    def __init__(
        self, 
        merge_mode: CategoricalValue("avg", "first"), # type: ignore
        batch_size = 4112,
        *, 
        verbose=True
    ):
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() and is_cuda_multiprocessing_enabled() else torch.device("cpu")
        )
        self.verbose = verbose
        self.print("Using device: %s" % self.device)
        self.merge_mode = merge_mode
        self.model = None
        self.tokenizer = None
        self.batch_size = batch_size
        
    def _merge(self, vectors):
        if not vectors.size(0):
            return np.zeros(vectors.size(1), dtype="float32")
        if self.merge_mode == "avg":
            return vectors.mean(dim=0).numpy()
        elif self.merge_mode == "first":
            return vectors[0, :].numpy()
        else:
            raise ValueError("Unknown merge mode")
    
    def run(self, input: Seq[Word]) -> MatrixContinuousDense:
        self.init_model()

        self.print("Tokenizing...", end="", flush=True)
        tokens = [self.tokenizer.tokenize(x) for x in input]
        sequence = self.tokenizer.encode_plus(
            [t for tokens in tokens for t in tokens], return_tensors="pt", padding=True, truncation=True,
        ).to(self.device)
        self.print("done")

        with torch.no_grad():
            self.print("Embedding...", end="", flush=True)
            output = self.model(**sequence).last_hidden_state
            output = output.squeeze(0)
            self.print("done")
            
        # delete the reference so we can clean the GRAM
        del sequence
        
        count = 0
        matrix = []
        for i, token in enumerate(input):
            contiguous = len(tokens[i])
            vectors = output[count : count + contiguous, :]
            vector = self._merge(vectors.to('cpu'))
            matrix.append(vector)
            count += contiguous
            
            # delete the reference so we can clean the GRAM
            del vectors

        matrix = np.vstack(matrix)
        torch.cuda.empty_cache()

        return matrix

class PretrainedSequenceEmbedding(TransformersWrapper):
    def __init__(
        self, 
        merge_mode: CategoricalValue("avg", "first"), # type: ignore
        batch_size = 4112,
        *, 
        verbose=True
    ):
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() and is_cuda_multiprocessing_enabled() else torch.device("cpu")
        )
        self.verbose = verbose
        self.print("Using device: %s" % self.device)
        self.merge_mode = merge_mode
        self.model = None
        self.tokenizer = None
        self.batch_size = batch_size
        
    def _merge(self, vectors):
        if not vectors.size(0):
            return np.zeros(vectors.size(1), dtype="float32")
        if self.merge_mode == "avg":
            return vectors.mean(dim=0).numpy()
        elif self.merge_mode == "first":
            return vectors[0, :].numpy()
        else:
            raise ValueError("Unknown merge mode")
    
    def run(self, input: Seq[Sentence]) -> MatrixContinuousDense:
        self.init_model()
                
        self.print("Tokenizing...", end="", flush=True)
        sequences = [self.tokenizer.encode_plus(
            sentence, return_tensors="pt", padding=True, truncation=True,
        ).to(self.device) for sentence in input]
        self.print("done")

        embeddings = []
        for i in tqdm(range(0, len(input), self.batch_size), desc="Processing batches"):
            batch = sequences[i:i+self.batch_size]

            with torch.no_grad():
                for bert_sequence in batch:
                    output = self.model(**bert_sequence).last_hidden_state
                    output = output.squeeze(0)

                    # Average the embeddings to get a sentence-level representation
                    sentence_embedding = output.mean(dim=0).to('cpu')
                    embeddings.append(sentence_embedding)

                    # delete the reference so we can clean the GRAM
                    del bert_sequence, output

            torch.cuda.empty_cache()

        # Stack the embeddings into a matrix
        embeddings_matrix = torch.stack(embeddings).numpy()
        return embeddings_matrix

class PretrainedTextGeneration(TransformersWrapper):
    def __init__(
        self, 
        merge_mode: CategoricalValue("avg", "first"), # type: ignore
        batch_size = 4112,
        *, 
        verbose=True
    ):
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() and is_cuda_multiprocessing_enabled() else torch.device("cpu")
        )
        self.verbose = verbose
        self.print("Using device: %s" % self.device)
        self.merge_mode = merge_mode
        self.model = None
        self.tokenizer = None
        self.batch_size = batch_size
        
    def _merge(self, vectors):
        if not vectors.size(0):
            return np.zeros(vectors.size(1), dtype="float32")
        if self.merge_mode == "avg":
            return vectors.mean(dim=0).numpy()
        elif self.merge_mode == "first":
            return vectors[0, :].numpy()
        else:
            raise ValueError("Unknown merge mode")
    
    def run(self, input: Seq[Word]) -> MatrixContinuousDense:
        self.init_model()

        self.print("Tokenizing...", end="", flush=True)
        tokens = [self.tokenizer.tokenize(x) for x in input]
        sequence = self.tokenizer.encode_plus(
            [t for tokens in tokens for t in tokens], return_tensors="pt", padding=True, truncation=True,
        ).to(self.device)
        self.print("done")

        with torch.no_grad():
            self.print("Embedding...", end="", flush=True)
            output = self.model(**sequence).last_hidden_state
            output = output.squeeze(0)
            self.print("done")
            
        # delete the reference so we can clean the GRAM
        del sequence
        
        count = 0
        matrix = []
        for i, token in enumerate(input):
            contiguous = len(tokens[i])
            vectors = output[count : count + contiguous, :]
            vector = self._merge(vectors.to('cpu'))
            matrix.append(vector)
            count += contiguous
            
            # delete the reference so we can clean the GRAM
            del vectors

        matrix = np.vstack(matrix)
        torch.cuda.empty_cache()

        return matrix

class PretrainedTextClassifier(TransformersWrapper):
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
        self.init_model()
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


def get_task_alias(task):
    for alias in TASK_ALIASES:
        if task in alias.value:
            return alias
    return None

TASK_TO_ALGORITHM_MARK = {
    TASK_ALIASES.TextClassification: "TEXT_CLASS_",
    TASK_ALIASES.WordEmbeddings: "WORD_EMB_",
    TASK_ALIASES.SeqEmbeddings: "SEQ_EMB_",
    TASK_ALIASES.TextGeneration: "TEXT_GEN_",
    TASK_ALIASES.TokenClassification: "TOKEN_CLASS_",
}

TASK_TO_WRAPPER_NAME = {
    TASK_ALIASES.WordEmbeddings: PretrainedWordEmbedding.__name__,
    TASK_ALIASES.SeqEmbeddings: PretrainedSequenceEmbedding.__name__,
    TASK_ALIASES.TextGeneration: PretrainedTextGeneration.__name__,
    TASK_ALIASES.TextClassification: PretrainedTextClassifier.__name__,
    TASK_ALIASES.TokenClassification: PretrainedTokenClassifier.__name__,
}

def build_transformers_wrappers(
    max_amount=1000, 
    min_likes=100, 
    min_downloads=1000,
    download_mode=DOWNLOAD_MODE.HUB
):
    path = Path(__file__).parent / "_generated.py"
    
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
        
        for target_task in TASK_ALIASES:
            manager = enlighten.get_manager()
            
            imports = download_models_info(
                target_task,
                max_amount=max_amount, 
                min_likes=min_likes, 
                min_downloads=min_downloads,
                download_mode=download_mode
            )
            
            counter = manager.counter(total=len(imports), unit="classes")

            for cls in imports:
                counter.update()
                _write_class(cls, fp, target_task)

            counter.close()
            manager.stop()
        
    black.reformat_one(
        path, True, black.WriteBack.YES, black.FileMode(), black.Report()
    )

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
            id2label = {item["metadata"]["id2label"]}
            num_classes = {len(item["metadata"]["id2label"])}
            tags = {len(item["metadata"]["id2label"])}
            model_type = "{item["metadata"]["model_type"]}"
            architectures = {item["metadata"]["architectures"]}
            vocab_size = {item["metadata"]["vocab_size"]}
            type_vocab_size = {item["metadata"]["type_vocab_size"]}
            is_decoder = {item["metadata"]["is_decoder"]}
            is_encoder_decoder = {item["metadata"]["is_encoder_decoder"]}
            num_layers = {item["metadata"]["num_layers"]}
            hidden_size = {item["metadata"]["hidden_size"]}
            num_attention_heads = {item["metadata"]["num_attention_heads"]}
            
            def __init__(
                self, batch_size: DiscreteValue(32, 256)
            ):
                {base_class}.__init__(self, batch_size)
        """
        )
    )

    print("Successfully generated" + class_name)
    fp.flush()


if __name__ == "__main__":
    import nltk
    nltk.download()
    
    build_transformers_wrappers(
        max_amount=100,
        download_mode=DOWNLOAD_MODE.BASE,
    )
    
    # build_transformers_wrappers(
    #     target_task=TASK_ALIASES.TokenClassification,
    #     download_file_path="token_classification_models_info.json",
    #     max_amount=20,
    #     download_mode=DOWNLOAD_MODE.SCRAP,
    #     min_likes=50
    # )
