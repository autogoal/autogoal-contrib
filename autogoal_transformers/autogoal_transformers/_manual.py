from autogoal.kb import (
    Label,
    Seq,
    Supervised,
    Word,
    Prompt,
    GeneratedText,
    Sentence,
    MatrixContinuousDense,
    VectorCategorical
)
from autogoal.kb._algorithm import _make_list_args_and_kwargs
from autogoal.kb import algorithm, AlgorithmBase
from autogoal.grammar import DiscreteValue, CategoricalValue, BooleanValue
from autogoal.utils import nice_repr
from autogoal_transformers._builder import TransformersWrapper
import numpy as np
import torch
from tqdm import tqdm
from autogoal.utils._process import is_cuda_multiprocessing_enabled
import textwrap
import re

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
        self.batch_size = 128#self.pretrained_text_generator.batch_size
        self.device = (
            torch.device("cuda")
            if torch.cuda.is_available() and is_cuda_multiprocessing_enabled()
            else torch.device("cpu")
        )

    def run(self, X: Seq[Sentence]) -> MatrixContinuousDense:
        self.pretrained_text_generator.init_model()
        embeddings_matrix = []
        for i in tqdm(range(0, len(X), self.batch_size)):
            batch_sentences = X[i:i+self.batch_size]
            inputs = self.pretrained_text_generator.tokenizer.batch_encode_plus(batch_sentences, return_tensors='pt', padding=True, truncation=True, max_length=512).to(self.device)

            with torch.no_grad():
                model = self.pretrained_text_generator.model.get_encoder() \
                    if self.pretrained_text_generator.is_encoder_decoder \
                    else self.pretrained_text_generator.model
                    
                outputs = model(**inputs, output_hidden_states=True)
                # outputs[0] contains the last hidden state
                batch_embeddings = outputs.hidden_states[-1].mean(dim=1).to("cpu").numpy()

            embeddings_matrix.extend(batch_embeddings)
            del inputs, outputs
            torch.cuda.empty_cache()
            
        return np.vstack(embeddings_matrix)

@nice_repr
class CARPClassifier(TransformersWrapper):
    def __init__(
        self,
        zero_shot: BooleanValue(), # type: ignore
        few_shots_amount: CategoricalValue(16, 32, 64, 128), # type: ignore
        training_examples_selection_method: CategoricalValue("random"), # type: ignore
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
        # if self.zero_shot:
        #     return y
        
        self.augmented_data = self.augment_data(X, y)
        return y

    def _eval(
        self, X, y
    ) -> VectorCategorical:
        self.init_model()
        
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
                training_examples = np.random.choice(range(len(self.augmented_data)), self.few_shots_amount)
            
            training_examples_text = "\n".join([self.augmented_data[i]["training_example"]  for i in training_examples])
            augmented_prompts = base_prompt + textwrap.dedent(f"""
                {training_examples_text}
                
                target
                INPUT: {X[i]}
                CLUES:
                REASONING:
                LABEL:
                """)
            
        generated_text = self.pretrained_text_generator.run(augmented_prompts)
        labels_pattern = "|".join(map(re.escape, self.unique_labels))
        pattern = f"(?<=target.*LABEL:.*?\\n)({labels_pattern})"
        
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
            augmented_data.append({
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
                """)
            })
            
        clues = self.pretrained_text_generator.run([x["clue_prompt"] for x in augmented_data])
        for i in range(len(augmented_data)):
            gen_clues = clues[i]
            prompt = augmented_data[i]["clue_prompt"]
            answer_index = clues[i].find(prompt) + len(prompt)
            
            if (answer_index >= 0):
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
                """)
        
        reasonings = self.pretrained_text_generator.run([x["reasoning_prompt"] for x in augmented_data])
        for i in range(len(augmented_data)):
            gen_reasoning = reasonings[i]
            prompt = augmented_data[i]["reasoning_prompt"]
            answer_index = clues[i].find(prompt) + len(prompt)
            
            if (answer_index >= 0):
                gen_reasoning = reasonings[i][answer_index:]
                
            augmented_data[i]["reasonings"] = gen_reasoning
            augmented_data[i]["reasoning_prompt"] = None
            augmented_data[i]["training_example"] = textwrap.dedent(f"""

                example {i}
                INPUT: {augmented_data[i]['sequence']} 
                CLUES: {augmented_data[i]['clues']} 
                REASONING: {augmented_data[i]['reasonings']}
                LABEL: {augmented_data[i]['label']}
                """)
        
        self.augmented_data = augmented_data
        self.unique_labels = unique_labels
        return augmented_data
    
    def run(
        self, X: Seq[Sentence], y: Supervised[VectorCategorical]
    ) -> VectorCategorical:
        return TransformersWrapper.run(self, X, y)
