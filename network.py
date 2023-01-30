from gensim.models import Word2Vec
from gensim.parsing.preprocessing import preprocess_string, remove_stopwords, strip_punctuation, strip_multiple_whitespaces
import html
import pandas as pd
from functions import BOOKS
    
def vectorizeSummary(summary: list, model) -> list:
    """Vectorizes words in a given summary and
    returns the list of the word vectors."""

    # Initialize vectorized words list
    vectors_list = []

    # Vectorize summary
    for word in summary:
        vectors_list.append(model.wv[word][0])
    
    return vectors_list

def vectorizeSummaries() -> pd.DataFrame:
    """Loads books from CSV into a DataFrame, vectorizes all
    books summaries and saves the new vector list in a new
    column of the books DataFrame, saving it into a new CSV."""

    # Initialize vectorized summaries list
    vect_summaries = []
    books = pd.read_csv(BOOKS)#.head(100)

    # Get all book summaries in a list
    summaries = books["summary"].to_list()

    # Pre-process summaries
    pre_proc_summaries = []
    for summary in summaries:
        # Pre-process summary, unescaping HTML characters, removing stop words etc
        summary = html.unescape(summary)
        CUSTOM_FILTERS = [
            lambda x: x.lower(), remove_stopwords,
            strip_multiple_whitespaces, strip_punctuation
        ]
        pre_proc_summaries.append(preprocess_string(summary, CUSTOM_FILTERS))
    
    print("Trainning new Word2Vec model...")
    model = Word2Vec(pre_proc_summaries, min_count=1, workers=6)
    print(f"Finished trainning model. Model summary: ", model, "\nVectorizing summaries...")

    # Iterate the books summaries
    for summary in pre_proc_summaries:
        vect_summaries.append(vectorizeSummary(summary, model))

    # Add vectorized summaries to the books DataFrame
    books["Vectorized_Summary"] = vect_summaries

    return books