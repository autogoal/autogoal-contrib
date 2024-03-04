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
from autogoal.kb import algorithm, AlgorithmBase
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
from transformers import get_linear_schedule_with_warmup

class SentenceDataset(Dataset):
    def __init__(self, sentences, labels, tokenizer, max_length):
        """
        Initializes the dataset.

        :param sentences: A list of sentences to be tokenized.
        :param tokenizer: The tokenizer to be used for tokenizing the sentences.
        :param max_length: The maximum length of the tokenized sequences.
        """
        self.sentences = sentences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = (
            torch.device("cuda:0")
            if torch.cuda.is_available() and is_cuda_multiprocessing_enabled()
            else torch.device("cpu")
        )

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        label = self.labels[idx] if not self.labels is None else None # Assuming labels are already integers
        inputs = self.tokenizer(
            sentence,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )
        
        # Flatten the output tensors to remove the batch dimension added by return_tensors
        inputs = {key: val.squeeze(0).to(self.device) for key, val in inputs.items()}
        
        if label is None:
            return {'data': inputs}
        return {'data': inputs, 'labels': torch.tensor(label, dtype=torch.long).to(self.device)}
    
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
class SequenceEmbeddingClassifier(nn.Module):
    def __init__(self, pretrained_embedder: PretrainedSequenceEmbedding, num_labels):
        super(SequenceEmbeddingClassifier, self).__init__()
        self.pretrained_embedder = pretrained_embedder
        self.device = (
            torch.device("cuda:0")
            if torch.cuda.is_available() and is_cuda_multiprocessing_enabled()
            else torch.device("cpu")
        )
        
        self.classifier =  nn.Sequential(
            nn.Linear(pretrained_embedder.embedding_dim, pretrained_embedder.embedding_dim),
            nn.Linear(pretrained_embedder.embedding_dim, num_labels),
            nn.Softmax()
        ).to(self.device)

    def forward(self, input_sentences):
        embeddings = self.pretrained_embedder.run_unbatched(input_sentences).to(self.device)
        logits = self.classifier(embeddings)
        return logits
    
@nice_repr
class FullFineTuner(AlgorithmBase):
    def __init__(
        self, 
        pretrained_model_seq_embedder: algorithm(Seq[Sentence], MatrixContinuousDense, exceptions=["DocumentEmbedder"]),   # type: ignore
        learning_rate: CategoricalValue(1e-5, 2e-5, 3e-5, 4e-5, 5e-5, 1e-4, 1e-6), # type: ignore
        epochs:DiscreteValue(1, 10), # type: ignore
    ):
        self.pretrained_model_seq_embedder = pretrained_model_seq_embedder
        self.learning_rate = 1e-5#learning_rate
        self.epochs = 5#epochs
        self.batch_size = 256#pretrained_model_seq_embedder.batch_size
        self.model = None
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
        
    def init_model(self, num_labels):
        self.pretrained_model_seq_embedder.init_model()
        self.model = SequenceEmbeddingClassifier(self.pretrained_model_seq_embedder, num_labels).to(self.device)

    def finetune(self, X, y):
        num_labels = len(np.unique(y))
        self.init_model(num_labels)
        
        dataset = SentenceDataset(X, y, self.pretrained_model_seq_embedder.tokenizer, max_length=512) #for BERT-like models
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        optimizer = AdamW(self.pretrained_model_seq_embedder.model.parameters(), lr=self.learning_rate)
        total_steps = len(dataloader) * self.epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

        self.model.train()
        for epoch in range(self.epochs):
            for batch in tqdm(dataloader):
                inputs = batch['data']
                labels = batch['labels']
                
                self.model.zero_grad()
                
                logits = self.model(inputs)
                
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, num_labels), labels.view(-1))
                
                loss.backward()
                optimizer.step()
                scheduler.step()

            print(f"Finished Epoch {epoch+1}, Loss: {loss.item()}")
        return y

    def predict(self, X):
        self.model.eval()
        
        dataset = SentenceDataset(X, None, self.pretrained_model_seq_embedder.tokenizer, max_length=512)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        
        predictions = []
        with torch.no_grad():
            for batch in tqdm(dataloader):
                inputs = batch['data']
                logits = self.model(inputs)
                predictions.extend(logits.argmax(dim=1).tolist())
        
        print("predicted", np.unique(predictions, return_counts=True))
        return predictions

    def run(self, X: Seq[Sentence], y: Supervised[VectorDiscrete]) -> VectorDiscrete:
        if (self._mode == "train"):
            return self.finetune(X, y)
        else:
            return self.predict(X)

    