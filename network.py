from gensim.models import Word2Vec
from gensim.parsing.preprocessing import preprocess_string, remove_stopwords, strip_punctuation, strip_multiple_whitespaces
import html
import pandas as pd
from functions import BOOKS
from os import cpu_count

# Threads to use when training Word2Vec model
TC = 8

def preProcessSummaries(summaries) -> list:
    """Pre-process summary, unescaping HTML characters,
    removing whitespaces, punctuations and stop words."""

    pre_proc_summaries = []
    
    for summary in summaries:
        summary = html.unescape(summary)
        CUSTOM_FILTERS = [
            lambda x: x.lower(), remove_stopwords,
            strip_multiple_whitespaces, strip_punctuation
        ]
        
        pre_proc_summaries.append(preprocess_string(summary, CUSTOM_FILTERS))

    return pre_proc_summaries
    
def replaceWordsWithVectors(summaries: list, model: Word2Vec) -> list:
    """Replaces words in summaries with their
    corresponsive vectors from a trained model."""

    vect_summaries = []

    for summary in summaries:
        vect_summary = []
        for word in summary:
            vect_summary.append(model.wv[word][0])
        vect_summaries.append(vect_summary)
    
    return vect_summaries

def vectorizeSummaries() -> pd.DataFrame:
    """Loads books from CSV into a DataFrame, vectorizes all
    books summaries and adds the new vector list in a new
    column of the books DataFrame, finally returning it."""

    # Load all book summaries from the CSV
    # in a list and preprocess them
    books = pd.read_csv(BOOKS)
    print("Pre-processing summaries...")
    pre_proc_summaries = preProcessSummaries(books["summary"].to_list())
    
    print("Training Word2Vec model...")
    workers = min(cpu_count(), TC)
    model = Word2Vec(pre_proc_summaries, min_count=1, workers=workers)
    print("Finished training model.")
    print(f"Model summary: {model}")

    print("Replacing words with vectors from trained model...")
    vect_summaries = replaceWordsWithVectors(pre_proc_summaries, model)

    # Add vectorized summaries to the books DataFrame
    books["Vectorized_Summary"] = vect_summaries

    return books