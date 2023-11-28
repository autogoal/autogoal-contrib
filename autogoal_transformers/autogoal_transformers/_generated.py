# AUTOGENERATED ON 2023-11-22 10:01:49.750997
## DO NOT MODIFY THIS FILE MANUALLY

from autogoal.utils import nice_repr
from numpy import inf, nan

from autogoal.grammar import (
    ContinuousValue,
    DiscreteValue,
    CategoricalValue,
    BooleanValue,
)
from autogoal_transformers._builder import (
    PretrainedZeroShotClassifier,
    PetrainedTextClassifier,
    PretrainedTokenClassifier,
)
from autogoal.kb import *

@nice_repr
class TEC_Facebook_BartLargeMnli(PretrainedZeroShotClassifier):
    name = "facebook/bart-large-mnli"
    likes = 742
    downloads = 2940000.0
    id2label = {"0": "contradiction", "1": "neutral", "2": "entailment"}
    num_classes = 3
    tags = 3

    def __init__(self, batch_size:DiscreteValue(128, 1024)):
        PretrainedZeroShotClassifier.__init__(self, batch_size)

@nice_repr
class TEC_Moritzlaurer_MdebertaV3BaseMnliXnli(PretrainedZeroShotClassifier):
    name = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
    likes = 167
    downloads = 58600.0
    id2label = {"0": "entailment", "1": "neutral", "2": "contradiction"}
    num_classes = 3
    tags = 3

    def __init__(self, batch_size:DiscreteValue(128, 1024)):
        PretrainedZeroShotClassifier.__init__(self, batch_size)

@nice_repr
class TEC_Moritzlaurer_MdebertaV3BaseXnliMultilingualNliMil7(
    PretrainedZeroShotClassifier
):
    name = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
    likes = 135
    downloads = 126000.0
    id2label = {"0": "entailment", "1": "neutral", "2": "contradiction"}
    num_classes = 3
    tags = 3

    def __init__(self, batch_size:DiscreteValue(128, 1024)):
        PretrainedZeroShotClassifier.__init__(self, batch_size)


@nice_repr
class TEC_Moritzlaurer_DebertaV3BaseMnliFeverAnli(PretrainedZeroShotClassifier):
    name = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
    likes = 110
    downloads = 7200000.0
    id2label = {"0": "entailment", "1": "neutral", "2": "contradiction"}
    num_classes = 3
    tags = 3

    def __init__(self, batch_size:DiscreteValue(128, 1024)):
        PretrainedZeroShotClassifier.__init__(self, batch_size)


@nice_repr
class TEC_Sileod_DebertaV3BaseTasksourceNli(PretrainedZeroShotClassifier):
    name = "sileod/deberta-v3-base-tasksource-nli"
    likes = 80
    downloads = 35700.0
    id2label = {"0": "entailment", "1": "neutral", "2": "contradiction"}
    num_classes = 3
    tags = 3

    def __init__(self, batch_size:DiscreteValue(128, 1024)):
        PretrainedZeroShotClassifier.__init__(self, batch_size)


@nice_repr
class TEC_Moritzlaurer_DebertaV3LargeMnliFeverAnliLingWanli(
    PretrainedZeroShotClassifier
):
    name = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
    likes = 62
    downloads = 9720.0
    id2label = {"0": "entailment", "1": "neutral", "2": "contradiction"}
    num_classes = 3
    tags = 3

    def __init__(self, batch_size:DiscreteValue(128, 1024)):
        PretrainedZeroShotClassifier.__init__(self, batch_size)


@nice_repr
class TEC_Valhalla_DistilbartMnli(PretrainedZeroShotClassifier):
    name = "valhalla/distilbart-mnli-12-1"
    likes = 40
    downloads = 40500.0
    id2label = {"0": "contradiction", "1": "neutral", "2": "entailment"}
    num_classes = 3
    tags = 3

    def __init__(self, batch_size:DiscreteValue(128, 1024)):
        PretrainedZeroShotClassifier.__init__(self, batch_size)


@nice_repr
class TEC_Moritzlaurer_DebertaV3BaseZeroshotV1(PretrainedZeroShotClassifier):
    name = "MoritzLaurer/deberta-v3-base-zeroshot-v1"
    likes = 34
    downloads = 7800.0
    id2label = {"0": "entailment", "1": "not_entailment"}
    num_classes = 2
    tags = 2

    def __init__(self, batch_size:DiscreteValue(128, 1024)):
        PretrainedZeroShotClassifier.__init__(self, batch_size)


@nice_repr
class TEC_Typeform_DistilbertBaseUncasedMnli(PretrainedZeroShotClassifier):
    name = "typeform/distilbert-base-uncased-mnli"
    likes = 33
    downloads = 32700.000000000004
    id2label = {"0": "ENTAILMENT", "1": "NEUTRAL", "2": "CONTRADICTION"}
    num_classes = 3
    tags = 3

    def __init__(self, batch_size:DiscreteValue(128, 1024)):
        PretrainedZeroShotClassifier.__init__(self, batch_size)


@nice_repr
class TEC_Moritzlaurer_MultilingualMinilmv2L6MnliXnli(PretrainedZeroShotClassifier):
    name = "MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli"
    likes = 23
    downloads = 9160.0
    id2label = {"0": "entailment", "1": "neutral", "2": "contradiction"}
    num_classes = 3
    tags = 3

    def __init__(self, batch_size:DiscreteValue(128, 1024)):
        PretrainedZeroShotClassifier.__init__(self, batch_size)


@nice_repr
class TEC_Vicgalle_XlmRobertaLargeXnliAnli(PretrainedZeroShotClassifier):
    name = "vicgalle/xlm-roberta-large-xnli-anli"
    likes = 22
    downloads = 39500.0
    id2label = {"0": "contradiction", "1": "neutral", "2": "entailment"}
    num_classes = 3
    tags = 3

    def __init__(self, batch_size:DiscreteValue(128, 1024)):
        PretrainedZeroShotClassifier.__init__(self, batch_size)


@nice_repr
class TEC_CrossEncoder_NliDistilrobertaBase(PretrainedZeroShotClassifier):
    name = "cross-encoder/nli-distilroberta-base"
    likes = 20
    downloads = 2970.0
    id2label = {"0": "contradiction", "1": "entailment", "2": "neutral"}
    num_classes = 3
    tags = 3

    def __init__(self, batch_size:DiscreteValue(128, 1024)):
        PretrainedZeroShotClassifier.__init__(self, batch_size)
