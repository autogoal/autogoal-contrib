from autogoal.datasets import semeval_2023_task_8_1 as semeval
from autogoal.datasets.semeval_2023_task_8_1 import F1_beta_plain, precision_plain, recall_plain, macro_f1_plain, macro_f1
from autogoal_sklearn._generated import MultinomialNB, MinMaxScaler, Perceptron
from autogoal_sklearn._manual import ClassifierTransformerTagger, ClassifierTagger
from autogoal_transformers._bert import BertEmbedding, BertTokenizeEmbedding
from autogoal_transformers._generated import TEC_Moritzlaurer_DebertaV3BaseMnliFeverAnli
from autogoal_transformers._tc_generated import TOC_Dslim_BertBaseNer
from autogoal_keras import KerasSequenceClassifier
from autogoal.kb import Seq, Word, VectorCategorical, MatrixCategorical, Supervised, Tensor, Categorical, Dense, Label, Pipeline, Sentence
from autogoal.datasets.meddocan import F1_beta, precision, recall
from autogoal.ml import AutoML, peak_ram_usage
from autogoal.search import RichLogger, NSPESearch
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
    X, y, X_test, y_test = semeval.load(mode=semeval.TaskTypeSemeval.TokenClassification, data_option=semeval.SemevalDatasetSelection.Original)
    
    a = AutoML(
        input=(Seq[Seq[Word]], Supervised[Seq[Seq[Label]]]),
        output=Seq[Seq[Label]],
        search_algorithm=NSPESearch,
        registry=find_classes(exclude="TEX"),#[BertEmbedding, ClassifierTagger, ClassifierTransformerTagger, Perceptron, MultinomialNB, MinMaxScaler, TOC_Dslim_BertBaseNer],#,#[BertEmbedding, ClassifierTagger, ClassifierTransformerTagger, Perceptron, MultinomialNB, MinMaxScaler, Arbert_RobertaBaseFinetunedNerKmeansTwitter],
        objectives=(macro_f1, peak_ram_usage),
        maximize=(True, False),
        evaluation_timeout=2*Min,
        pop_size=10,
        memory_limit=20*Gb,
        search_timeout=5*Min
    )
    
    amount = 10

    X_train = X[:amount]
    y_train = y[:amount]
    
    X_test = X_test[:amount]
    y_test = y_test[:amount]
    
    loggers = [RichLogger()]
    a.fit(X_train, y_train, logger=loggers)
    
    results = a.predict(X_test)
    print(f"F1: {macro_f1(y_test, results)}, precision: {precision(y_test, results)}, recall: {recall(y_test, results)}")

def test_semeval_sentence_classification():
    X, y, _, _ = semeval.load(mode=semeval.TaskTypeSemeval.SentenceClassification, data_option=semeval.SemevalDatasetSelection.Original, classes_mapping=semeval.TargetClassesMapping.Original)

    a = AutoML(
        input=(Seq[Sentence], Supervised[VectorCategorical]),
        output=VectorCategorical,
        registry=find_classes(include="TEC"),
        objectives=(macro_f1_plain, peak_ram_usage),
        maximize=(True, False),
        evaluation_timeout=5*Min,
        search_timeout=10*Min,
        memory_limit=20*Gb
    )
    
    amount = 20
    
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
    test_semeval_token_classification()
    # test_semeval_sentence_classification()
    
    # X, y, _, _ = semeval.load(mode=semeval.TaskTypeSemeval.TokenClassification, data_option=semeval.SemevalDatasetSelection.Original)
    # import numpy as np
    # from deap import base, creator, tools, algorithms
    # import random
    # import sys
    
    # # print(len(y))
    # # print(len(selected_y))
    
    # # X_train, y_train, X_test, y_test = semeval.load(mode=semeval.TaskTypeSemeval.TokenClassification, data_option=semeval.SemevalDatasetSelection.Original)
    # # print(len(X_train)/len(X_test))
    
    
    # classes = ['O', 'claim', 'per_exp', 'claim_per_exp', 'question']
    
    # original_list = []
    # for yi in y:
    #     counts = tuple(yi.count(cls) for cls in classes)
    #     counts = tuple(count / sum(counts) for count in counts)
    #     original_list.append(counts)
        
    # def get_proportions(list):
    #     # Calculate the sum of each component across all tuples
    #     component_sums = [sum(tup[i] for tup in list) for i in range(len(list[0]))]
    #     # Calculate the total sum of all components
    #     total_sum = sum(component_sums)
    #     # Calculate the proportion of each component
        
    #     try:
    #         return [comp_sum / total_sum for comp_sum in component_sums]
    #     except ZeroDivisionError:
    #         return [0 for comp_sum in component_sums]
    
    # def euclidean_distance(a, b):
    #     return np.sqrt(np.sum((np.array(a) - np.array(b))**2))

    # def func(original_list, selected_indexes):
    #     if len(selected_indexes) == 0:
    #         return sys.float_info.max
    
    #     notselected = [original_list[i] for i in range(len(original_list)) if i not in selected_indexes]
    #     selected = [original_list[i] for i in selected_indexes]
        
    #     notselected_proportions = get_proportions(notselected)
    #     selected_proportions = get_proportions(selected)
        
    #     return euclidean_distance(notselected_proportions, selected_proportions)
        
        
    # # Define the fitness function
    # creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    # creator.create("Individual", list, fitness=creator.FitnessMin)

    # toolbox = base.Toolbox()
    
    # # Attribute generator 
    # toolbox.register("index", random.randint, 0, len(original_list)-1)

    # # Structure initializers
    # toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.index, n=int(len(original_list)*0.2))
    # toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # def evalFunc(individual):
    #     return func(original_list, individual),

    # # Operator registering
    # toolbox.register("evaluate", evalFunc)
    # toolbox.register("mate", tools.cxTwoPoint)
    # toolbox.register("mutate", tools.mutUniformInt, low=0, up=len(original_list)-1, indpb=0.05)
    # toolbox.register("select", tools.selTournament, tournsize=3)

    # def main():
    #     pop = toolbox.population(n=50)
    #     hof = tools.HallOfFame(1)
    #     stats = tools.Statistics(lambda ind: ind.fitness.values)
    #     stats.register("avg", np.mean)
    #     stats.register("min", np.min)
    #     stats.register("max", np.max)
        
    #     pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=10, stats=stats, halloffame=hof, verbose=True)\
        
    #     best_indexes = hof[0]
    #     best_list = [original_list[i] for i in best_indexes]
        
    #     with open("best_indexes.txt", "w") as f:
    #         import json
    #         json.dump(best_indexes, f)
        
    #     return pop, logbook, hof
    
    # print(main())
