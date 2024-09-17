from autogoal.kb import (
    Label,
    Seq,
    Supervised,
    Word,
    Prompt,
    GeneratedText,
    Sentence,
    MatrixContinuousDense,
    VectorCategorical,
    VectorDiscrete,
    Document,
)
from autogoal.kb._algorithm import _make_list_args_and_kwargs
from autogoal.kb import algorithm, AlgorithmBase, VectorContinuous
from autogoal.grammar import DiscreteValue, CategoricalValue, BooleanValue, ContinuousValue
from autogoal.utils import nice_repr
from autogoal_transformers._builder import TransformersWrapper, PretrainedSequenceEmbedding
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from autogoal.utils._process import is_cuda_multiprocessing_enabled
import textwrap
import re

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup, get_constant_schedule_with_warmup, AutoTokenizer, AutoModelForSequenceClassification
from autogoal_transformers._utils import SimpleTextDataset
from peft import get_peft_model, LoraConfig, TaskType
import os

@nice_repr
class SeqPretrainedTokenClassifier(AlgorithmBase):
    def __init__(
        self,
        pretrained_token_classifier: algorithm(Seq[Word], Supervised[Seq[Label]], Seq[Label]),  # type: ignore
    ) -> None:
        super().__init__()
        self.inner = pretrained_token_classifier

    def run(self, X: Seq[Seq[Word]], y: Supervised[Seq[Seq[Label]]]) -> Seq[Seq[Label]]:
        args_kwargs = _make_list_args_and_kwargs(X, y)
        return [self.inner.run(*t.args, **t.kwargs) for t in args_kwargs]

@nice_repr
class TGenerationBasedPretrainedEmbedder(AlgorithmBase):
    def __init__(
        self,
        pretrained_text_generator: algorithm(Seq[Prompt], Seq[GeneratedText]),  # type: ignore
    ) -> None:
        super().__init__()
        self.pretrained_text_generator = pretrained_text_generator
        self.batch_size = 128  # self.pretrained_text_generator.batch_size
        self.device = (
            torch.device("cuda")
            if torch.cuda.is_available() and is_cuda_multiprocessing_enabled()
            else torch.device("cpu")
        )

    def run(self, X: Seq[Sentence]) -> MatrixContinuousDense:
        self.pretrained_text_generator.init_model()
        embeddings_matrix = []
        for i in tqdm(range(0, len(X), self.batch_size)):
            batch_sentences = X[i : i + self.batch_size]
            inputs = self.pretrained_text_generator.tokenizer.batch_encode_plus(
                batch_sentences,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(self.device)

            with torch.no_grad():
                model = (
                    self.pretrained_text_generator.model.get_encoder()
                    if self.pretrained_text_generator.is_encoder_decoder
                    else self.pretrained_text_generator.model
                )

                outputs = model(**inputs, output_hidden_states=True)
                # outputs[0] contains the last hidden state
                batch_embeddings = (
                    outputs.hidden_states[-1].mean(dim=1).to("cpu").numpy()
                )

            embeddings_matrix.extend(batch_embeddings)
            del inputs, outputs
            torch.cuda.empty_cache()
        return np.vstack(embeddings_matrix)

@nice_repr
class CARPClassifier(TransformersWrapper):
    def __init__(
        self,
        few_shots_amount: CategoricalValue(2, 4, 8, 16, 32, 64, 128),  # type: ignore
        training_examples_selection_method: CategoricalValue("random"),  # type: ignore
        pretrained_text_generator: algorithm(Seq[Prompt], Seq[GeneratedText]),  # type: ignore
    ) -> None:
        super().__init__()
        self.pretrained_text_generator = pretrained_text_generator
        self.batch_size = self.pretrained_text_generator.batch_size
        self.few_shots_amount = few_shots_amount
        self.training_examples_selection_method = training_examples_selection_method
        self.device = (
            torch.device("cuda")
            if torch.cuda.is_available() and is_cuda_multiprocessing_enabled()
            else torch.device("cpu")
        )

    def _train(self, X, y):
        self.pretrained_text_generator.init_model()
        self.pretrained_text_generator.max_gen_seq_length = 200
        self.pretrained_text_generator.temperature = 1

        self.augmented_data = self.augment_data(X, y)
        return y

    def _eval(self, X, y) -> VectorCategorical:
        self.pretrained_text_generator.init_model()

        base_prompt = textwrap.dedent(
            f"""
            This is a text classifier. Only respond with the text to complete and nothing more at all.
            
            List CLUES (i.e., keywords, phrases, contextual information, semantic meaning, semantic relationships, tones, references) that support the class determination of the input. 
            Next, deduce the diagnostic REASONING process from premises (i.e., clues, input) that support the class determination. 
            Finally, based on clues, the reasoning and the input, categorize the overall classof input as one of the following: {unique_labels_text}.
            
            Your answer should be the CLUES, REASONING and later the LABEL for the target. Make sure you base your response on the examples below.
            """
        )

        unique_labels_text = ", ".join(self.unique_labels)
        augmented_prompts = []
        for i in range(len(X)):
            training_examples = []
            if self.training_examples_selection_method == "random":
                training_examples = np.random.choice(
                    range(len(self.augmented_data)), self.few_shots_amount
                )

            training_examples_text = "\n".join(
                [self.augmented_data[i]["training_example"] for i in training_examples]
            )
            augmented_prompts.append(base_prompt + textwrap.dedent(
                f"""
                {training_examples_text}
                
                target
                INPUT: {X[i]}
                CLUES:
                REASONING:
                LABEL:
                """
            ))

        generated_text = self.pretrained_text_generator.run(augmented_prompts)
        labels_pattern = "|".join(map(re.escape, self.unique_labels))
        pattern = f"{labels_pattern}"

        results = []
        for text in generated_text:
            matches = re.findall(pattern, text, re.DOTALL)
            if matches:
                results.append(matches[-1])
            else:
                results.append(np.random.choice(self.unique_labels))

    def augment_data(self, X, y):
        unique_labels = np.unique(y)
        unique_labels_text = ", ".join(unique_labels)

        # prepare clue prompts
        augmented_data = []
        for sequence, label in zip(X, y):
            augmented_data.append(
                {
                    "sequence": sequence,
                    "label": label,
                    "clue_prompt": textwrap.dedent(
                        f"""
                    Only respond with the text to complete and nothing more at all.
                    This is a generic text classifier. The possible classes are: {unique_labels_text}.
                    
                    Complete the CLUES (i.e., keywords, phrases, contextual information, semantic meaning, semantic relationships, tones, references) that support the label determination of the input (limit to 15 words).
                    Your response should be a list of words or phrases separated by commas and should be right after "CLUES:".
                    
                    INPUT: {sequence}
                    GOLD LABEL: {label}
                    CLUES: 
                """
                    ),
                }
            )

        clues = self.pretrained_text_generator.run(
            [x["clue_prompt"] for x in augmented_data]
        )
        for i in range(len(augmented_data)):
            gen_clues = clues[i]
            prompt = augmented_data[i]["clue_prompt"]
            answer_index = clues[i].find(prompt) + len(prompt)

            if answer_index >= 0:
                gen_clues = clues[i][answer_index:]

            augmented_data[i]["clues"] = gen_clues
            augmented_data[i]["clue_prompt"] = None
            augmented_data[i]["reasoning_prompt"] = textwrap.dedent(
                f"""
                    Only respond with the text to complete and nothing more at all.
                    This is a generic text classifier. The possible classes are: {unique_labels_text}.
                    
                    Based on the input and clues, articulate the diagnostic reasoning process that supports the class (label) determination of the input.
                    Your response should be a reasoning text and should be right after "REASONING:".
                    
                    INPUT: {sequence}
                    LABEL: {label}
                    CLUES: {clues[i]}
                    REASONING:
                """
            )

        reasonings = self.pretrained_text_generator.run(
            [x["reasoning_prompt"] for x in augmented_data]
        )
        for i in range(len(augmented_data)):
            gen_reasoning = reasonings[i]
            prompt = augmented_data[i]["reasoning_prompt"]
            answer_index = clues[i].find(prompt) + len(prompt)

            if answer_index >= 0:
                gen_reasoning = reasonings[i][answer_index:]

            augmented_data[i]["reasonings"] = gen_reasoning
            augmented_data[i]["reasoning_prompt"] = None
            augmented_data[i]["training_example"] = textwrap.dedent(
                f"""

                example {i}
                INPUT: {augmented_data[i]['sequence']} 
                CLUES: {augmented_data[i]['clues']} 
                REASONING: {augmented_data[i]['reasonings']}
                LABEL: {augmented_data[i]['label']}
                """
            )

        self.augmented_data = augmented_data
        self.unique_labels = unique_labels
        return augmented_data

    def run(
        self, X: Seq[Sentence], y: Supervised[VectorCategorical]
    ) -> VectorCategorical:
        return TransformersWrapper.run(self, X, y)

@nice_repr
class GenerativeClassifier(TransformersWrapper):
    def __init__(
        self,
        zero_shot: BooleanValue(),  # type: ignore
        few_shots_amount: CategoricalValue(16, 32, 64, 128),  # type: ignore
        training_examples_selection_method: CategoricalValue("random"),  # type: ignore
        pretrained_text_generator: algorithm(Seq[Prompt], Seq[GeneratedText]),  # type: ignore
    ) -> None:
        super().__init__()
        self.pretrained_text_generator = pretrained_text_generator
        self.batch_size = self.pretrained_text_generator.batch_size
        self.zero_shot = zero_shot
        self.few_shots_amount = few_shots_amount
        self.training_examples_selection_method = training_examples_selection_method
        self.device = (
            torch.device("cuda")
            if torch.cuda.is_available() and is_cuda_multiprocessing_enabled()
            else torch.device("cpu")
        )

    def _train(self, X, y):
        self.pretrained_text_generator.init_model()
        self.pretrained_text_generator.max_gen_seq_length = 200
        self.pretrained_text_generator.temperature = 1
        self.unique_labels = np.unique(y)
        self.unique_labels_text = ", ".join(self.unique_labels)
        
        if self.zero_shot:
            return y

        self.augmented_data = list(zip(X, y))
        return y

    def _eval(self, X, y) -> VectorCategorical:
        self.pretrained_text_generator.init_model()
        base_prompt = textwrap.dedent(
            f"""
            This is a text classifier. Only respond with the target class to predict. Follow the next steps for arriving to a target LABEL.
            Categorize the target class of input as one of the following: {self.unique_labels_text}.
            
            Make sure you base your response on the examples below. RESPOND ONLY WITH THE LABEL!!!.
            """
        )

        augmented_prompts = []
        if (not self.zero_shot):
            for i in range(len(X)):
                training_examples = []
                if self.training_examples_selection_method == "random":
                    training_examples = np.random.choice(
                        range(len(self.augmented_data)), self.few_shots_amount
                    )

                training_examples_text = "\n".join([
                    f"""
                    example {i}
                    INPUT: {self.augmented_data[i][0]}
                    LABEL: {self.augmented_data[i][1]}
                    """ for i in training_examples]
                ) 
                
                nprompt = base_prompt + textwrap.dedent(
                    f"""
                    {training_examples_text}
                    
                    target
                    INPUT: {X[i]}
                    LABEL:
                    """
                )
                augmented_prompts.append(nprompt)
            else:
                augmented_prompts = [base_prompt + textwrap.dedent(
                    f"""
                    target
                    INPUT: {X[i]}
                    LABEL:
                    """
                ) for i in range(len(X))]

        generated_text = self.pretrained_text_generator.run(augmented_prompts)
        labels_pattern = "|".join(map(re.escape, self.unique_labels))
        pattern = f"{labels_pattern}"

        results = []
        for text in generated_text:
            matches = re.findall(pattern, text, re.DOTALL)
            if matches:
                results.append(matches[-1])
            else:
                results.append(np.random.choice(self.unique_labels))
        return results

    def run(
        self, X: Seq[Sentence], y: Supervised[VectorCategorical]
    ) -> VectorCategorical:
        return TransformersWrapper.run(self, X, y)

@nice_repr
class DocumentEmbedder(AlgorithmBase):
    def __init__(
        self,
        seq_embedder: algorithm(Seq[Sentence], MatrixContinuousDense, exceptions=["DocumentEmbedder"]),  # type: ignore
        sent_tokenizer: algorithm(Document, Seq[Sentence]),  # type: ignore
        pooling: CategoricalValue("mean", "max", "rms"),  # type: ignore
        normalization_strategy: CategoricalValue("l2", "l1", "min-max", "z-score", "none"),  # type: ignore
    ) -> None:
        super().__init__()
        self.seq_embedder = seq_embedder
        self.sent_tokenizer = sent_tokenizer
        self.pooling = pooling
        self.normalization_strategy = normalization_strategy
        self.device = (
            torch.device("cuda:0")
            if torch.cuda.is_available() and is_cuda_multiprocessing_enabled()
            else torch.device("cpu")
        )

    def run(self, X: Seq[Document]) -> MatrixContinuousDense:
        all_sentences = []  # To store all sentences from all documents
        doc_to_sent_indices = []  # To track which sentences belong to which document

        for doc in X:
            sentences = self.sent_tokenizer.run(
                doc
            )  # Assuming this returns a list of sentences
            all_sentences.extend(sentences)
            doc_to_sent_indices.append(len(sentences))

        # Step 2: Embed sentences
        sentence_embeddings = self.seq_embedder.run(all_sentences)

        # Step 3: Group embeddings by document and apply pooling
        doc_embeddings = []
        start_idx = 0
        for num_sentences in doc_to_sent_indices:
            end_idx = start_idx + num_sentences
            doc_sent_embeddings = sentence_embeddings[start_idx:end_idx]

            doc_embeddings.append(self.pool(doc_sent_embeddings))
            start_idx = end_idx

        if self.normalization_strategy != "none":
            doc_embeddings = self.normalize(torch.stack(doc_embeddings))
            
        return doc_embeddings

    def pool(self, doc_sent_embeddings):
        doc_sent_embeddings_tensors = [torch.tensor(emb, dtype=torch.float) if isinstance(emb, np.ndarray) else emb for emb in doc_sent_embeddings]
        stacked_embeddings = torch.stack(doc_sent_embeddings_tensors).to(self.device)
        
        if self.pooling == "mean":
            doc_embedding = torch.mean(stacked_embeddings, dim=0)
        elif self.pooling == "max":
            doc_embedding, _ = torch.max(stacked_embeddings, dim=0)
        elif self.pooling == "rms":
            doc_embedding = torch.sqrt(torch.mean(stacked_embeddings**2, dim=0))
        else:
            raise ValueError("Unsupported pooling method")
        return doc_embedding.to("cpu")

    def normalize(self, embeddings):
        if self.normalization_strategy == "l2":
            normalized_embeddings = F.normalize(embeddings, p=2, dim=1)
        elif self.normalization_strategy == "l1":
            normalized_embeddings = F.normalize(embeddings, p=1, dim=1)
        elif self.normalization_strategy == "min-max":
            min_val = embeddings.min(dim=1, keepdim=True)[0]
            max_val = embeddings.max(dim=1, keepdim=True)[0]
            normalized_embeddings = (embeddings - min_val) / (max_val - min_val)
        elif self.normalization_strategy == "z-score":
            mean = embeddings.mean(dim=1, keepdim=True)
            std = embeddings.std(dim=1, keepdim=True)
            normalized_embeddings = (embeddings - mean) / std
        elif self.normalization_strategy == "none":
            normalized_embeddings = embeddings  # No normalization applied
        else:
            raise ValueError(
                f"Unknown normalization strategy: {self.normalization_strategy}"
            )
        return normalized_embeddings.to("cpu")

@nice_repr
class FullFineTunerBase(AlgorithmBase):
    def __init__(
        self, 
    ):
        self._mode = "train"
        self.device = (
            torch.device("cuda:0")
            if torch.cuda.is_available() and is_cuda_multiprocessing_enabled()
            else torch.device("cpu")
        )
        
    def train(self):
        self._mode = "train"

    def eval(self):
        self._mode = "eval"
        
    def init_model(self, model, num_labels):
        model.init_model(
            transformer_model_cls=AutoModelForSequenceClassification, 
            tokenizer_cls=AutoTokenizer, 
            transformer_cls_kwargs={
                'num_labels':num_labels,
                'trust_remote_code': True
            }
        )
        self.model = model.model.to(self.device)
        self.tokenizer = model.tokenizer
        
        if self.tokenizer.pad_token is None:
            print("No padding token. Adding EOS as PAD token.")
            self.tokenizer.pad_token = self.tokenizer.eos_token  # Setting pad_token as eos_token for generative models
            self.model.config.pad_token_id = self.tokenizer.pad_token
            
    def setup_optimizer(self):
        # Set up the optimizer
        if self.optimizer == 'adamw':
            return AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.optimizer == 'adam':
            return torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.optimizer == 'sgd':
            return torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

    def setup_scheduler(self, optimizer, total_steps):
        # Set up the learning rate scheduler
        if self.lr_scheduler == 'linear':
            return get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.warmup_steps, num_training_steps=total_steps)
        elif self.lr_scheduler == 'cosine':
            return get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=self.warmup_steps, num_training_steps=total_steps)
        elif self.lr_scheduler == 'constant':
            return get_constant_schedule_with_warmup(optimizer, num_warmup_steps=self.warmup_steps)

    def setup_dropout(self):
        # Set the dropout rate if necessary
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.p = self.dropout_rate
                
    def finetune(self, X, y):
        pass

    def predict(self, X):
        dataset = SimpleTextDataset(X, None, self.tokenizer, max_length=self.max_length)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        self.model.eval()
        preds = []
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                inputs = {key: val.to(self.device) for key, val in batch.items() if key != 'labels'}
                outputs = self.model(**inputs)
                logits = outputs.logits
                preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
        return preds
    
    def run(self, X: Seq[Sentence], y: Supervised[VectorDiscrete]) -> VectorDiscrete:
        if (self._mode == "train"):
            return self.finetune(X, y)
        else:
            return self.predict(X)

@nice_repr
class FullFineTunerTextTransformerClassifier(FullFineTunerBase):
    def __init__(
        self, 
        inner_model: algorithm(*[Word, VectorContinuous], include=["transformer"]), # type: ignore
        batch_size: DiscreteValue(32, 512),  # type: ignore
        max_length: DiscreteValue(32, 512), # type: ignore
        learning_rate: CategoricalValue(1e-5, 2e-5, 3e-5, 4e-5, 5e-5, 1e-4, 1e-6), # type: ignore
        epochs: DiscreteValue(1, 5), # type: ignore
        warmup_steps: DiscreteValue(0, 2000), # type: ignore
        weight_decay: CategoricalValue(0, 0.01, 0.1), # type: ignore
        dropout_rate: CategoricalValue(0.1, 0.2, 0.3, 0.4, 0.5), # type: ignore
        optimizer: CategoricalValue('adamw', 'adam', 'sgd'), # type: ignore
        gradient_accumulation_steps: DiscreteValue(1, 8), # type: ignore
        lr_scheduler: CategoricalValue('linear', 'cosine', 'constant') # type: ignore
    ):
        self.model = None
        self.tokenizer = None
        self.inner_model = inner_model
        self.batch_size = batch_size
        self.max_length = max_length
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay
        self.dropout_rate = dropout_rate
        self.optimizer = optimizer
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.lr_scheduler = lr_scheduler
        super().__init__()
        
    def finetune(self, X, y):
        num_labels = len(np.unique(y))
        self.init_model(self.inner_model, num_labels)
        self.setup_dropout()
        
        dataset = SimpleTextDataset(X, y, self.tokenizer, max_length=self.max_length) #for BERT-like models
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        optimizer = self.setup_optimizer()#AdamW(self.pretrained_model_seq_embedder.model.parameters(), lr=self.learning_rate)
        total_steps = len(dataloader) * self.epochs
        scheduler = self.setup_scheduler(optimizer, total_steps)#get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0
            for step, batch in enumerate(tqdm(dataloader, desc="Training")):
                optimizer.zero_grad()
                inputs = {key: val.to(self.device) for key, val in batch.items() if key != 'labels'}
                labels = batch['labels'].to(self.device)
                outputs = self.model(**inputs, labels=labels)
                loss = outputs.loss / self.gradient_accumulation_steps
                loss.backward()
                
                if (step + 1) % self.gradient_accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                total_loss += loss.item()
            
            print(f"Epoch {epoch+1}/{self.epochs}, Training Loss: {total_loss / len(dataloader)}")

        return y

@nice_repr
class PartialFineTunerTextTransformerClassifier(FullFineTunerBase):
    def __init__(
        self, 
        inner_model: algorithm(*[Word, VectorContinuous], include=["transformer"]), # type: ignore
        num_trainable_layers: DiscreteValue(0, 50),  # type: ignore
        batch_size: DiscreteValue(32, 512),  # type: ignore
        max_length: DiscreteValue(32, 512), # type: ignore
        learning_rate: CategoricalValue(1e-5, 2e-5, 3e-5, 4e-5, 5e-5, 1e-4, 1e-6), # type: ignore
        epochs: DiscreteValue(1, 5), # type: ignore
        warmup_steps: DiscreteValue(0, 2000), # type: ignore
        weight_decay: CategoricalValue(0, 0.01, 0.1), # type: ignore
        dropout_rate: CategoricalValue(0.1, 0.2, 0.3, 0.4, 0.5), # type: ignore
        optimizer: CategoricalValue('adamw', 'adam', 'sgd'), # type: ignore
        gradient_accumulation_steps: DiscreteValue(1, 8), # type: ignore
        lr_scheduler: CategoricalValue('linear', 'cosine', 'constant') # type: ignore
    ):
        self.model = None
        self.tokenizer = None
        self.inner_model = inner_model
        self.num_trainable_layers = num_trainable_layers
        self.batch_size = batch_size
        self.max_length = max_length
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay
        self.dropout_rate = dropout_rate
        self.optimizer = optimizer
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.lr_scheduler = lr_scheduler
        super().__init__()
        
    def count_trainable_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def set_freezed_layers(self):
        # Unfreeze the specified number of layers from end to start
        layer_groups = self.group_layers_by_number()
        
        # Freeze all layers initially
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Unfreeze classifier layers and optionally other layers
        classifier_params = [p for n, p in self.model.named_parameters() if 'classifier' in n]
        for param in classifier_params:
            param.requires_grad = True

        # Unfreeze additional layers based on the number of layers to train
        layers_unfreezed = 0
        total_layers = len(layer_groups)
        for layer_num in sorted(layer_groups.keys(), reverse=True):
            if layers_unfreezed < self.num_trainable_layers:
                for param in layer_groups[layer_num]:
                    param.requires_grad = True
                layers_unfreezed += 1
            else:
                break
        
        # Optionally handle embeddings and other non-numbered layers
        if layers_unfreezed < self.num_trainable_layers:
            non_layer_params = [p for n, p in self.model.named_parameters() if not re.search(r'\.layer\.\d+\.', n) and 'classifier' not in n]
            for param in non_layer_params:
                param.requires_grad = True

    def group_layers_by_number(self):
        # Group parameters by their layer number
        layer_groups = {}
        pattern = re.compile(r'\.layer\.(\d+)\.')
        for name, param in self.model.named_parameters():
            match = pattern.search(name)
            if match:
                layer_num = int(match.group(1))
                if layer_num not in layer_groups:
                    layer_groups[layer_num] = []
                layer_groups[layer_num].append(param)
        return layer_groups

    def finetune(self, X, y):
        num_labels = len(np.unique(y))
        self.init_model(self.inner_model, num_labels)
        self.setup_dropout()
        self.set_freezed_layers()
        
        dataset = SimpleTextDataset(X, y, self.tokenizer, max_length=self.max_length) #for BERT-like models
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        optimizer = self.setup_optimizer()#AdamW(self.pretrained_model_seq_embedder.model.parameters(), lr=self.learning_rate)
        total_steps = len(dataloader) * self.epochs
        scheduler = self.setup_scheduler(optimizer, total_steps)#get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

        print(f"trainable parameters: {self.count_trainable_parameters()}")

        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0
            for step, batch in enumerate(tqdm(dataloader, desc="Training")):
                optimizer.zero_grad()
                inputs = {key: val.to(self.device) for key, val in batch.items() if key != 'labels'}
                labels = batch['labels'].to(self.device)
                outputs = self.model(**inputs, labels=labels)
                loss = outputs.loss / self.gradient_accumulation_steps
                loss.backward()
                
                if (step + 1) % self.gradient_accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                total_loss += loss.item()
            
            print(f"Epoch {epoch+1}/{self.epochs}, Training Loss: {total_loss / len(dataloader)}")
        return y
    
@nice_repr
class LORAFineTunerTextTransformerClassifier(FullFineTunerBase):
    def __init__(
        self, 
        inner_model: algorithm(*[Word, VectorContinuous], include=["transformer"]), # type: ignore
        lora_r: CategoricalValue(2, 4, 8, 16), # type: ignore
        lora_alpha: CategoricalValue(8, 16, 32, 64), # type: ignore
        lora_dropout: CategoricalValue(0.0, 0.1, 0.2, 0.3), # type: ignore
        lora_bias: CategoricalValue("none", "all"), # type: ignore
        batch_size: DiscreteValue(32, 512),  # type: ignore
        max_length: DiscreteValue(32, 512), # type: ignore
        learning_rate: CategoricalValue(1e-5, 2e-5, 3e-5, 4e-5, 5e-5, 1e-4, 1e-6), # type: ignore
        epochs: DiscreteValue(1, 10), # type: ignore
        warmup_steps: DiscreteValue(0, 2000), # type: ignore
        weight_decay: CategoricalValue(0, 0.01, 0.1), # type: ignore
        optimizer: CategoricalValue('adamw', 'adam', 'sgd'), # type: ignore
        gradient_accumulation_steps: DiscreteValue(1, 8), # type: ignore
        lr_scheduler: CategoricalValue('linear', 'cosine', 'constant') # type: ignore
    ):
        self.model = None
        self.tokenizer = None
        self.inner_model = inner_model
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.lora_bias = lora_bias
        self.batch_size = batch_size
        self.max_length = max_length
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay
        self.optimizer = optimizer
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.lr_scheduler = lr_scheduler
        super().__init__()
        
    def set_lora_config(self):
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout, 
            bias=self.lora_bias,
            modules_to_save=["classifier"],
        )
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
    
    def print_model_layers(self):
        for name, param in self.model.named_parameters():
            print(name, param.shape)    
    
    def finetune(self, X, y):
        num_labels = len(np.unique(y))
        self.init_model(self.inner_model, num_labels)
        self.set_lora_config()
        
        dataset = SimpleTextDataset(X, y, self.tokenizer, max_length=self.max_length) #for BERT-like models
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        optimizer = self.setup_optimizer()#AdamW(self.pretrained_model_seq_embedder.model.parameters(), lr=self.learning_rate)
        total_steps = len(dataloader) * self.epochs
        scheduler = self.setup_scheduler(optimizer, total_steps)#get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0
            for step, batch in enumerate(tqdm(dataloader, desc="Training")):
                optimizer.zero_grad()
                inputs = {key: val.to(self.device) for key, val in batch.items() if key != 'labels'}
                labels = batch['labels'].to(self.device)
                outputs = self.model(**inputs, labels=labels)
                loss = outputs.loss / self.gradient_accumulation_steps
                loss.backward()
                
                if (step + 1) % self.gradient_accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                total_loss += loss.item()
            
            print(f"Epoch {epoch+1}/{self.epochs}, Training Loss: {total_loss / len(dataloader)}")

        return y
    
@nice_repr
class FullFineTunerGenerativeTransformerClassifier(FullFineTunerTextTransformerClassifier):
    def __init__(
        self, 
        inner_model: algorithm(*[Prompt, GeneratedText], include=["transformer"]), # type: ignore
        batch_size: DiscreteValue(32, 512),  # type: ignore
        max_length: DiscreteValue(32, 512), # type: ignore
        learning_rate: CategoricalValue(1e-5, 2e-5, 3e-5, 4e-5, 5e-5, 1e-4, 1e-6), # type: ignore
        epochs: DiscreteValue(1, 5), # type: ignore
        warmup_steps: DiscreteValue(0, 2000), # type: ignore
        weight_decay: CategoricalValue(0, 0.01, 0.1), # type: ignore
        dropout_rate: CategoricalValue(0.1, 0.2, 0.3, 0.4, 0.5), # type: ignore
        optimizer: CategoricalValue('adamw', 'adam', 'sgd'), # type: ignore
        gradient_accumulation_steps: DiscreteValue(1, 8), # type: ignore
        lr_scheduler: CategoricalValue('linear', 'cosine', 'constant') # type: ignore
    ):
        super().__init__(
            inner_model, 
            batch_size, 
            max_length, 
            learning_rate, 
            epochs, 
            warmup_steps, 
            weight_decay, 
            dropout_rate, 
            optimizer, 
            gradient_accumulation_steps, 
            lr_scheduler)

@nice_repr
class PartialFineTunerGenerativeTransformerClassifier(PartialFineTunerTextTransformerClassifier):
    def __init__(
        self, 
        inner_model: algorithm(*[Prompt, GeneratedText], include=["transformer"]), # type: ignore
        num_trainable_layers: DiscreteValue(0, 50),  # type: ignore
        batch_size: DiscreteValue(32, 512),  # type: ignore
        max_length: DiscreteValue(32, 512), # type: ignore
        learning_rate: CategoricalValue(1e-5, 2e-5, 3e-5, 4e-5, 5e-5, 1e-4, 1e-6), # type: ignore
        epochs: DiscreteValue(1, 5), # type: ignore
        warmup_steps: DiscreteValue(0, 2000), # type: ignore
        weight_decay: CategoricalValue(0, 0.01, 0.1), # type: ignore
        dropout_rate: CategoricalValue(0.1, 0.2, 0.3, 0.4, 0.5), # type: ignore
        optimizer: CategoricalValue('adamw', 'adam', 'sgd'), # type: ignore
        gradient_accumulation_steps: DiscreteValue(1, 8), # type: ignore
        lr_scheduler: CategoricalValue('linear', 'cosine', 'constant') # type: ignore
    ):
        super().__init__(
            inner_model,
            num_trainable_layers,
            batch_size, 
            max_length, 
            learning_rate, 
            epochs, 
            warmup_steps, 
            weight_decay, 
            dropout_rate, 
            optimizer, 
            gradient_accumulation_steps, 
            lr_scheduler)
        
@nice_repr
class LORAFineTunerGenerativeTransformerClassifier(LORAFineTunerTextTransformerClassifier):
    def __init__(
        self, 
        inner_model: algorithm(*[Prompt, GeneratedText], include=["transformer"]), # type: ignore
        lora_r: CategoricalValue(2, 4, 8, 16), # type: ignore
        lora_alpha: CategoricalValue(8, 16, 32, 64), # type: ignore
        lora_dropout: CategoricalValue(0.0, 0.1, 0.2, 0.3), # type: ignore
        lora_bias: CategoricalValue("none", "all"), # type: ignore
        batch_size: DiscreteValue(32, 512),  # type: ignore
        max_length: DiscreteValue(32, 512), # type: ignore
        learning_rate: CategoricalValue(1e-5, 2e-5, 3e-5, 4e-5, 5e-5, 1e-4, 1e-6), # type: ignore
        epochs: DiscreteValue(1, 10), # type: ignore
        warmup_steps: DiscreteValue(0, 2000), # type: ignore
        weight_decay: CategoricalValue(0, 0.01, 0.1), # type: ignore
        optimizer: CategoricalValue('adamw', 'adam', 'sgd'), # type: ignore
        gradient_accumulation_steps: DiscreteValue(1, 8), # type: ignore
        lr_scheduler: CategoricalValue('linear', 'cosine', 'constant') # type: ignore
    ):
        super().__init__(
            inner_model,
            lora_r,
            lora_alpha,
            lora_dropout,
            lora_bias,
            batch_size, 
            max_length, 
            learning_rate, 
            epochs, 
            warmup_steps, 
            weight_decay, 
            optimizer, 
            gradient_accumulation_steps, 
            lr_scheduler)


    def __init__(
        self, 
        inner_model: algorithm(*[Prompt, GeneratedText], include=["transformer"]), # type: ignore
        lora_r: CategoricalValue(2, 4, 8, 16), # type: ignore
        lora_alpha: CategoricalValue(8, 16, 32, 64), # type: ignore
        lora_dropout: CategoricalValue(0.0, 0.1, 0.2, 0.3), # type: ignore
        lora_bias: CategoricalValue("none", "all"), # type: ignore
        batch_size: DiscreteValue(32, 512),  # type: ignore
        max_length: DiscreteValue(32, 512), # type: ignore
        learning_rate: CategoricalValue(1e-5, 2e-5, 3e-5, 4e-5, 5e-5, 1e-4, 1e-6), # type: ignore
        epochs: DiscreteValue(1, 10), # type: ignore
        warmup_steps: DiscreteValue(0, 2000), # type: ignore
        weight_decay: CategoricalValue(0, 0.01, 0.1), # type: ignore
        optimizer: CategoricalValue('adamw', 'adam', 'sgd'), # type: ignore
        gradient_accumulation_steps: DiscreteValue(1, 8), # type: ignore
        lr_scheduler: CategoricalValue('linear', 'cosine', 'constant') # type: ignore
    ):
        super().__init__(
            inner_model,
            lora_r,
            lora_alpha,
            lora_dropout,
            lora_bias,
            batch_size, 
            max_length, 
            learning_rate, 
            epochs, 
            warmup_steps, 
            weight_decay, 
            optimizer, 
            gradient_accumulation_steps, 
            lr_scheduler)