from autogoal_transformers._builder import TransformersWrapper
from autogoal.kb import (Label, Seq, Supervised,
                         Word)
from autogoal.kb._algorithm import _make_list_args_and_kwargs
from autogoal.kb import algorithm, AlgorithmBase


class SeqPretrainedTokenClassifier(AlgorithmBase):
    def __init__(
        self, 
        pretrained_token_classifier: algorithm(Seq[Word], Supervised[Seq[Label]], Seq[Label]),
        ) -> None:
        super().__init__()
        self.inner = pretrained_token_classifier
        
    def run(self, X: Seq[Seq[Word]], y: Supervised[Seq[Seq[Label]]]) -> Seq[Seq[Label]]:
        args_kwargs = _make_list_args_and_kwargs(X, y)
        return [self.inner.run(*t.args, **t.kwargs) for t in args_kwargs]
    