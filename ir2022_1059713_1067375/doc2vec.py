import pandas as pd
from functions import BOOKS
import html
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.parsing.preprocessing import preprocess_string, remove_stopwords, strip_punctuation, strip_multiple_whitespaces
from pprint import pprint
from os import cpu_count
from gensim.utils import tokenize

# Threads to use when training Doc2Vec model
TC = 12
PROCESSED_FILES = "Files/Processed/"
BOOKS_VECTORIZED_SUMMARIES_PKL = PROCESSED_FILES + "Books-Vectorized-Summaries.pkl"
D2V_MODEL = PROCESSED_FILES + "Doc2VecModel.mod"

def preProcessSummary(summary) -> list:
    """Pre-process summary, unescaping HTML characters,
    removing whitespaces, punctuations and stop words."""
    
    summary = html.unescape(summary)
    CUSTOM_FILTERS = [
        lambda x: x.lower(), remove_stopwords,
        strip_multiple_whitespaces, strip_punctuation
    ]

    return preprocess_string(summary, CUSTOM_FILTERS)

def preProcessSummaryv2(summary) -> list:
    return list(tokenize(html.unescape(summary), lowercase=True, deacc=True))

def trainDoc2VecModel(books: pd.DataFrame) -> Doc2Vec:
    summaries = books["summary"].to_list()

    # Pre-process all summaries
    preproc_sums = [preProcessSummaryv2(summary) for summary in summaries]
    tagged_data = [TaggedDocument(d, [i]) for i, d in enumerate(preproc_sums)]

    threads = min(cpu_count(), TC)
    model = Doc2Vec(tagged_data, min_count = 1, epochs = 10, workers=threads)
    return model

def vectorizeSummaries(books_to_vect: pd.DataFrame, trainning_books: pd.DataFrame = None) -> pd.DataFrame:
    """Vectorize book summaries and add them in a new column of the books dataframe."""
    
    doc2vecModel = askForPreload(trainning_books)

    print("Vectorizing books...")
    doc_vectors = []
    summaries = books_to_vect["summary"].to_list()
    for summary in summaries:
        preproc_sum = preProcessSummaryv2(summary)
        doc_vectors.append(doc2vecModel.infer_vector(preproc_sum))
    books_to_vect["Vectorized_Summary"] = doc_vectors

    return books_to_vect

def askForPreload(training_books) -> Doc2Vec:
    while True:
        pre_load = input("Try loading pre-trained Doc2Vec model? (y/n): ")
        if pre_load == "y":
            try:
                model = Doc2Vec.load(D2V_MODEL)
                print("Loaded pre-trained Doc2Vec model.")
                return model
            except:
                print("Model couldn't be loaded.\nTraining Doc2Vec model...")
                break
        elif pre_load == "n":
            print("Training Doc2Vec model...")
            break
    
    model = trainDoc2VecModel(training_books)
    model.save(D2V_MODEL)

    return model

#print("Turning summaries into vectors...")
#books_with_vecsums = vectorizeSummaries(books, model)
#print("Saving vectorized summaries...")
#books_with_vecsums.to_pickle(BOOKS_VECTORIZED_SUMMARIES_PKL)