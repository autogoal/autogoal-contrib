from autogoal_contrib import find_classes
from autogoal_transformers import BertEmbedding, BertTokenizeEmbedding

BertTokenizeEmbedding.download()

for alg in find_classes("TOC"):
    try:
        alg.download()
    except:
        print(f"Failed to download {alg.__name__}")