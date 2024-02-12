try:
    import gensim
except:
    print("(!) Code in `autogoal_gensim` requires `gensim`.")
    print("(!) You can install it with `pip install autogoal-[gensim]`.")
    raise


from autogoal_gensim._base import (
    Word2VecEmbedding,
    Word2VecEmbeddingSpanish,
    FastTextEmbeddingSpanishSUC,
    FastTextEmbeddingSpanishSWBC,
    GloveEmbeddingSpanishSWBC,
)
