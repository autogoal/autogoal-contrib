from autogoal.datasets import semeval_2023_task_8_1 as semeval
from autogoal.datasets.semeval_2023_task_8_1 import F1_beta_plain, precision_plain, recall_plain, macro_f1_plain
from autogoal_sklearn._generated import MultinomialNB, MinMaxScaler, Perceptron
from autogoal_sklearn._manual import ClassifierTransformerTagger, ClassifierTagger
from autogoal_transformers._bert import BertEmbedding, BertTokenizeEmbedding
from autogoal_keras import KerasSequenceClassifier
from autogoal.kb import Seq, Word, VectorCategorical, MatrixCategorical, Supervised, Tensor, Categorical, Dense, Label, Pipeline, Sentence
from autogoal.datasets.meddocan import F1_beta, precision, recall
from autogoal.ml import AutoML
from autogoal.search import RichLogger
from autogoal_telegram import TelegramLogger
from autogoal.utils import Gb, Min

from autogoal_contrib import find_classes

def test_pipeline(data_size = 1000, opt = False):
    X, y, _, _ = semeval.load(mode=semeval.TaskTypeSemeval.TokenClassification, data_option=semeval.SemevalDatasetSelection.Actual)
    
    X_train = X[:data_size]
    y_train = y[:data_size]
    
    X_test = X[data_size:2*data_size]
    y_test = y[data_size:2*data_size]
    
    pipeline = Pipeline(algorithms=
        [
            ClassifierTagger(classifier=Perceptron(
                    l1_ratio=0.15,
                    fit_intercept=True,
                    tol=0.001,
                    shuffle=True,
                    eta0=1,
                    early_stopping=False,
                    validation_fraction=0.1,
                    n_iter_no_change=5
                )) if opt else ClassifierTransformerTagger(transformer=MinMaxScaler(clip=True), classifier=MultinomialNB(fit_prior=False)),
        ],
        input_types=[Seq[Seq[Word]], Supervised[Seq[VectorCategorical]]],
    )
    
    pipeline.send("send")
    pipeline.run(X_train, y_train)
    
    pipeline.send("eval")
    predicted = pipeline.run(X_test, y_test)
    
    print(f"results: F1_beta {F1_beta(y_test, predicted)}, Precision {precision(y_test, predicted)}, Recall {recall(y_test, predicted)}")
    
def test_semeval_token_classification():
    X, y, _, _ = semeval.load(mode=semeval.TaskTypeSemeval.TokenClassification, data_option=semeval.SemevalDatasetSelection.Original)

    a = AutoML(
        input=(Seq[Seq[Word]], Supervised[Seq[Seq[Label]]]),
        output=Seq[Seq[Label]],
        registry=find_classes(exclude="Keras"),#[BertEmbedding, ClassifierTagger, ClassifierTransformerTagger, Perceptron, MultinomialNB, MinMaxScaler, Arbert_RobertaBaseFinetunedNerKmeansTwitter],
        objectives=F1_beta,
        evaluation_timeout=10*Min,
        memory_limit=20*Gb
    )
    
    amount = 200

    X_train = X[:amount]
    y_train = y[:amount]
    
    X_test = X[amount:2*amount]
    y_test = y[amount:2*amount]
    
    loggers = [RichLogger(), TelegramLogger(token="6425450979:AAF4Mic12nAWYlfiMNkCTRB0ZzcgaIegd7M")]
    a.fit(X_train, y_train, logger=loggers)
    
    results = a.predict(X_test)
    print(f"F1: {F1_beta(y_test, results)}, precision: {precision(y_test, results)}, recall: {recall(y_test, results)}")

def test_semeval_sentence_classification():
    X, y, _, _ = semeval.load(mode=semeval.TaskTypeSemeval.SentenceClassification, data_option=semeval.SemevalDatasetSelection.Original, classes_mapping=semeval.TargetClassesMapping.Extended)

    a = AutoML(
        input=(Seq[Sentence], Supervised[VectorCategorical]),
        output=VectorCategorical,
        registry=find_classes(include="TEC"),
        objectives=macro_f1_plain,
        evaluation_timeout=5*Min,
        search_timeout=10*Min,
        memory_limit=20*Gb
    )
    
    amount = 100
    
    X_train = X[:amount]
    y_train = y[:amount]
    
    X_test = X[amount:2*amount]
    y_test = y[amount:2*amount]
    
    loggers = [RichLogger(), TelegramLogger(token="6425450979:AAF4Mic12nAWYlfiMNkCTRB0ZzcgaIegd7M", channel="570734906", name="test")]
    a.fit(X_train, y_train, logger=loggers)
    
    results = a.score(X_test, y_test)
    print(f"F1: {results}")

if __name__ == '__main__':
    
    # BertTokenizeEmbedding.download()
    # BertEmbedding.download()
    
    # test_pipeline(3000, True)
    # test_pipeline(3000)
    # from autogoal.utils._process import initialize_cuda_multiprocessing
    
    # initialize_cuda_multiprocessing()
    # test_semeval_token_classification()
    test_semeval_sentence_classification()