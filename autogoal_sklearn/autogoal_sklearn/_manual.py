from sklearn.feature_extraction.text import CountVectorizer as _CountVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn_crfsuite import CRF

from autogoal_sklearn._builder import SklearnTransformer, SklearnEstimator
from autogoal.kb import *
from autogoal.grammar import (
    BooleanValue,
    CategoricalValue,
    DiscreteValue,
    ContinuousValue,
)
from autogoal.utils import nice_repr
from autogoal.kb import AlgorithmBase, Supervised
import numpy as np

@nice_repr
class CountVectorizerTokenizeStem(_CountVectorizer, SklearnTransformer):
    def __init__(
        self,
        lowercase: BooleanValue(),
        stopwords_remove: BooleanValue(),
        binary: BooleanValue(),
        inner_tokenizer: algorithm(Sentence, Seq[Word]),
        inner_stemmer: algorithm(Word, Stem),
        inner_stopwords: algorithm(Seq[Word], Seq[Word]),
    ):
        self.stopwords_remove = stopwords_remove
        self.inner_tokenizer = inner_tokenizer
        self.inner_stemmer = inner_stemmer
        self.inner_stopwords = inner_stopwords

        SklearnTransformer.__init__(self)
        _CountVectorizer.__init__(self, lowercase=lowercase, binary=binary)

    def build_tokenizer(self):
        def func(sentence):
            tokens = self.inner_tokenizer.run(sentence)
            tokens = (
                self.inner_stopwords.run(sentence) if self.stopwords_remove else tokens
            )
            return [self.inner_stemmer.run(token) for token in tokens]

        return func

    def run(self, input: Seq[Sentence]) -> MatrixContinuousSparse:
        return SklearnTransformer.run(self, input)


class _FeatureVectorizer(SklearnTransformer):
    def __init__(self, sparse):
        self.vectorizer = DictVectorizer(sparse=sparse)
        super().__init__()

    def fit_transform(self, X, y=None):
        return self.vectorizer.fit_transform(X)

    def transform(self, X, y=None):
        return self.vectorizer.transform(X, y=y)


@nice_repr
class FeatureSparseVectorizer(_FeatureVectorizer):
    def __init__(self):
        super().__init__(sparse=True)

    def run(self, input: Seq[FeatureSet]) -> MatrixContinuousSparse:
        return super().run(input)


@nice_repr
class FeatureDenseVectorizer(_FeatureVectorizer):
    def __init__(self):
        super().__init__(sparse=False)

    def run(self, input: Seq[FeatureSet]) -> MatrixContinuousDense:
        return super().run(input)


@nice_repr
class CRFTagger(CRF, SklearnEstimator):
    def __init__(
        self, algorithm: CategoricalValue("lbfgs", "l2sgd", "ap", "pa", "arow")
    ) -> None:
        SklearnEstimator.__init__(self)
        super().__init__(algorithm=algorithm)

    def run(
        self, X: Seq[Seq[FeatureSet]], y: Supervised[Seq[Seq[Label]]]
    ) -> Seq[Seq[Label]]:
        return SklearnEstimator.run(self, X, y)


@nice_repr
class ClassifierTagger(SklearnEstimator):
    """
    A wrapper class that uses a classifier as a tagger. This class is designed to work with classifiers, 
    possibly from sklearn, and use them as taggers. It handles the necessary transformations to fit and predict on sequence data.

    Attributes:
        classifier (algorithm): The classifier algorithm to use. 
        This should be a function that takes a MatrixContinuous and a Supervised[VectorCategorical] as input and returns a VectorCategorical.
    """
    def __init__(
        self, 
        classifier: algorithm(MatrixContinuous, Supervised[VectorCategorical], VectorCategorical)
    ) -> None:
        SklearnEstimator.__init__(self)
        self.classifier = classifier
        
    def fit(self, X, y):
        # Save the starting index of each sublist in X
        self.concat_positions = [0] + [len(sublist) if isinstance(sublist, list) else sublist.shape[0] for sublist in X]
        self.concat_positions = np.cumsum(self.concat_positions).tolist()[:-1]

        # Concatenate X and y into single lists
        X_concat = [embedding for sublist in X for embedding in sublist]
        y_concat = [label for sublist in y for label in sublist]
        
        # Fit the classifier
        self.classifier.fit(X_concat, y_concat)
        return y

    def predict(self, X):
        # Concatenate X into a single list of embeddings
        X_concat = [embedding for sublist in X for embedding in sublist]
        
        # Predict using the classifier
        y_pred_concat = self.classifier.predict(X_concat)
        
        # Deconcatenate y_pred to match the original shape of X
        y_pred = [np.array([y_pred_concat[y_pos + i] for i in range(len(sub_X) if isinstance(sub_X, list) else sub_X.shape[0])]) for y_pos, sub_X in enumerate(X)]
            
        # Check the shapes of the predictions agains the original elements
        for i in range(len(X)):
            sy = np.shape(y_pred[i])
            sx = np.shape(X[i])
            assert sy[0] == sx[0]

        return y_pred

    def run(
        self, X: Seq[MatrixContinuous], y: Supervised[Seq[Seq[Label]]]
    ) -> Seq[Seq[Label]]:
        return SklearnEstimator.run(self, X, y)

@nice_repr
class ClassifierTransformerTagger(SklearnEstimator):
    """
    A wrapper class that uses a classifier as a tagger. This class is designed to work with classifiers, 
    possibly from sklearn, and use them as taggers. It handles the necessary transformations to fit and predict on sequence data.

    Attributes:
        classifier (algorithm): The classifier algorithm to use. 
        This should be a function that takes a MatrixContinuous and a Supervised[VectorCategorical] as input and returns a VectorCategorical.
    """
    def __init__(
        self, 
        transformer: algorithm(MatrixContinuousDense, MatrixContinuousDense),
        classifier: algorithm(MatrixContinuous, Supervised[VectorCategorical], VectorCategorical)
    ) -> None:
        SklearnEstimator.__init__(self)
        self.classifier = classifier
        self.transformer = transformer
        
    def fit(self, X, y):
        # Save the starting index of each sublist in X
        self.concat_positions = [0] + [len(sublist) if isinstance(sublist, list) else sublist.shape[0] for sublist in X]
        self.concat_positions = np.cumsum(self.concat_positions).tolist()[:-1]

        # Concatenate X and y into single lists
        X_concat = [embedding for sublist in X for embedding in sublist]
        y_concat = [label for sublist in y for label in sublist]
        
        if (self.transformer):
            X_concat = self.transformer.fit_transform(X_concat)
            
        # Fit the classifier
        self.classifier.fit(X_concat, y_concat)
        return y

    def predict(self, X):
        # Concatenate X into a single list of embeddings
        X_concat = [embedding for sublist in X for embedding in sublist]

        if (self.transformer):
            X_concat = self.transformer.fit_transform(X_concat)

        # Predict using the classifier
        y_pred_concat = self.classifier.predict(X_concat)

        # Deconcatenate y_pred to match the original shape of X
        y_pred = [np.array([y_pred_concat[y_pos + i] for i in range(len(sub_X) if isinstance(sub_X, list) else sub_X.shape[0])]) for y_pos, sub_X in enumerate(X)]

        return y_pred

    def run(
        self, X: Seq[MatrixContinuous], y: Supervised[Seq[Seq[Label]]]
    ) -> Seq[Seq[Label]]:
        return SklearnEstimator.run(self, X, y)

@nice_repr
class AggregatedTransformer(SklearnTransformer):
    def __init__(
        self, 
        transformer: algorithm(MatrixContinuousDense, MatrixContinuousDense)
    ) -> None:
        SklearnTransformer.__init__(self)
        self.transformer = transformer
    
    def fit_transform(self, X, y=None):
        # Save the starting index of each sublist in X
        self.concat_positions = [0] + [len(sublist) if isinstance(sublist, list) else sublist.shape[0] for sublist in X]
        self.concat_positions = np.cumsum(self.concat_positions).tolist()[:-1]
        
        # Concatenate X and y into single lists
        X_concat = [embedding for sublist in X for embedding in sublist]
        
        if hasattr(self, "partial_fit"):
            self.partial_fit(X_concat)
            result = self.transformer.transform(X_concat)
        else:
            result = self.transformer.fit_transform(X_concat)
            
        # Deconcatenate y_pred to match the original shape of X
        return [np.array([result[y_pos + i] for i in range(len(sub_X) if isinstance(sub_X, list) else sub_X.shape[0])]) for y_pos, sub_X in enumerate(X)]

    def transform(self, X, y=None):
        # Save the starting index of each sublist in X
        self.concat_positions = [0] + [len(sublist) if isinstance(sublist, list) else sublist.shape[0] for sublist in X]
        self.concat_positions = np.cumsum(self.concat_positions).tolist()[:-1]
        
        # Concatenate X and y into single lists
        X_concat = [embedding for sublist in X for embedding in sublist]
        
        result = self.transformer.transform(X_concat)
            
        # Deconcatenate y_pred to match the original shape of X
        return [np.array([result[y_pos + i] for i in range(len(sub_X) if isinstance(sub_X, list) else sub_X.shape[0])]) for y_pos, sub_X in enumerate(X)]
        
    def run(self, X: Seq[MatrixContinuous]) -> Seq[MatrixContinuous]:
        return SklearnTransformer.run(self, X)

__all__ = [
    "CountVectorizerTokenizeStem",
    "FeatureSparseVectorizer",
    "FeatureDenseVectorizer",
    "CRFTagger",
    "ClassifierTagger",
    "ClassifierTransformerTagger"
]
