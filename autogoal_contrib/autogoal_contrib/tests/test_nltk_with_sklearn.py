from autogoal_contrib import find_classes
from autogoal.utils import is_package_installed
from autogoal.kb import (
    algorithm,
    Sentence,
    Seq,
    Word,
    Stem,
    build_pipeline_graph,
    Supervised,
    Label,
    Categorical,
    Dense,
    Pipeline,
    Tensor,
    Postag,
)
import pytest
from autogoal.grammar import generate_cfg, Symbol


class Algorithm:
    def __init__(
        self,
        tokenizer: algorithm(Sentence, Seq[Word]),
        stemmer: algorithm(Word, Stem),
        stopword: algorithm(Seq[Word], Seq[Word]),
    ) -> None:
        self.tokenizer = tokenizer
        self.stemmer = stemmer
        self.stopword = stopword


@pytest.mark.skipif(
    not is_package_installed("autogoal_nltk"), reason="The test requires autogoal_nltk"
)
def test_find_nltk_implementations():
    grammar = generate_cfg(Algorithm, find_classes(include=["*.nltk.*"]))

    assert Symbol("Algorithm[[Sentence],Seq[Word]]") in grammar._productions
    assert Symbol("Algorithm[[Word],Stem]") in grammar._productions
    assert Symbol("Algorithm[[Seq[Word]],Seq[Word]]") in grammar._productions


@pytest.mark.skipif(
    not is_package_installed("autogoal_sklearn")
    or not is_package_installed("autogoal_nltk"),
    reason="The test requires autogoal_sklearn and autogoal_nltk",
)
def test_crf_pipeline():
    from autogoal_nltk import FeatureSeqExtractor
    from autogoal_sklearn import CRFTagger

    graph = build_pipeline_graph(
        input_types=(Seq[Seq[Word]], Supervised[Seq[Seq[Label]]]),
        output_type=Seq[Seq[Label]],
        registry=[FeatureSeqExtractor, CRFTagger],
    )

    pipeline = graph.sample()


@pytest.mark.skipif(
    not is_package_installed("autogoal_sklearn")
    or not is_package_installed("autogoal_nltk"),
    reason="The test requires autogoal_sklearn and autogoal_nltk",
)
def test_count_vectorizer_sgd():
    from autogoal_sklearn import CountVectorizer
    from autogoal_sklearn._generated import SGDClassifier

    p = Pipeline(
        algorithms=[
            CountVectorizer(lowercase=True, binary=True),
            SGDClassifier(
                loss="perceptron",
                penalty="l1",
                l1_ratio=0.999,
                fit_intercept=False,
                tol=0.001,
                shuffle=True,
                epsilon=0.24792790326293826,
                learning_rate="optimal",
                eta0=0.992,
                power_t=4.991,
                early_stopping=False,
                validation_fraction=0.993,
                n_iter_no_change=1,
                average=True,
            ),
        ],
        input_types=(Seq[Sentence], Supervised[Tensor[1, Categorical, Dense]]),
    )

    Xtrain = ["hello world", "this is sparta"]
    ytrain = ["true", "false"]

    p.run(Xtrain, ytrain)
    p.send("eval")
    p.run(Xtrain, None)


@pytest.mark.skipif(
    not is_package_installed("autogoal_sklearn")
    or not is_package_installed("autogoal_nltk"),
    reason="The test requires autogoal_sklearn and autogoal_nltk",
)
def test_classifier_tagger():
    from autogoal_nltk._generated import ClassifierBasedPOSTagger

    p = Pipeline(
        algorithms=[
            ClassifierBasedPOSTagger(),
        ],
        input_types=(Seq[Seq[Word]], Supervised[Seq[Seq[Postag]]]),
    )

    p.run([["hello", "world"]], [["A", "B"]])

    p.send("eval")

    result = p.run([["hello", "world"]], None)
    assert result == [["A", "B"]]
