import autogoal_transformers
from autogoal_transformers._generated import Jiva_XlmRobertaLargeItMnli
from autogoal_transformers._tc_generated import Jiva_XlmRobertaLargeItMnli
from autogoal.datasets import imdb_50k_movie_reviews
from sklearn.metrics import accuracy_score

X, y = imdb_50k_movie_reviews.load()

a = Jiva_XlmRobertaLargeItMnli()

a.train()
a.run(X[:25000], y[:25000])

amount = 2000

a.eval()
result = a.run(X[:amount], y[:amount])
for t in range(amount):
    print(f"{t} - {X[t]} got {result[t]} expected {y[t]}")
print("Done")

print(f"Accuracy: {accuracy_score(result, y[:amount])}")