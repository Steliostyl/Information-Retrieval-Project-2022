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
    model = Doc2Vec(tagged_data, min_count = 1, epochs = 10, workers=threads, vector_size=512)
    return model

def vectorizeSummaries(books_to_vect: pd.DataFrame, doc2vecModel: Doc2Vec) -> pd.DataFrame:
    """Vectorize book summaries and add them in a new column of the books dataframe."""

    doc_vectors = []
    summaries = books_to_vect["summary"].to_list()
    for summary in summaries:
        preproc_sum = preProcessSummaryv2(summary)
        doc_vectors.append(doc2vecModel.infer_vector(preproc_sum))
    books_to_vect["Vectorized_Summary"] = doc_vectors

    return books_to_vect

def getDoc2VecModel(training_books: pd.DataFrame = None) -> Doc2Vec:
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
        
def getVectorizedBooks(books_to_vect:pd.DataFrame, doc2VecModel: Doc2Vec, books: pd.DataFrame = None) -> pd.DataFrame:
    while True:
        pre_load = input("Load vectorized books from file? (y/n): ")
        if pre_load == "y":
            try:
                vectorized_books = pd.read_pickle(BOOKS_VECTORIZED_SUMMARIES_PKL)
                vectorized_es_books = pd.merge(books_to_vect, vectorized_books, on="isbn")
                print("Loaded vectorized books from file.")
                return vectorized_es_books
            except:
                print("Couldn't find vectorized books file.")
                break
        elif pre_load == "n":
            break
    print("Vectorizing elastic books' summaries...")

    vectorized_es_books = vectorizeSummaries(books_to_vect=books_to_vect, doc2vecModel=doc2VecModel)

    print("Finished vectorizing elastic books' summaries.")
    if type(books) is None:
        return vectorized_es_books
    
    while True:
        pre_load = input("Vectorize all books and save them to file? (y/n): ")
        if pre_load == "y":
            df = pd.merge(vectorized_es_books, books, on="isbn", how="outer", indicator=True)
            unvectorized_books = df[df["_merge"] == "right_only"]
            vect_unvect_books = vectorizeSummaries(unvectorized_books, books)
            all_vectorized_books = pd.concat([vectorized_es_books, vect_unvect_books], ignore_index=True)
            all_vectorized_books.to_pickle(BOOKS_VECTORIZED_SUMMARIES_PKL)
            break
        elif pre_load == "n":
            break
        
    return vectorized_es_books

def run():
    elastic_books = pd.read_csv("Files/Processed/Elastic_Books.csv")
    books = pd.read_csv("Files/Input/BX-Books.csv")
    avg_clust_ratings = pd.read_csv("Files/Processed/Average-Cluster-Rating.csv")

    # Get average cluster ratings of user's cluster
    users_clust_avg_ratings = avg_clust_ratings.loc[avg_clust_ratings["cluster"] == 0]
    df = pd.merge(elastic_books, users_clust_avg_ratings, on=['isbn'], how="outer", indicator=True)
    # Get books in es_books but not in users_clust_avg_ratings
    rel_unrated_books = df[df['_merge'] == 'left_only']
    
    doc2vecModel = getDoc2VecModel(books)
    vectorized_rel_unrated_books = getVectorizedBooks(rel_unrated_books, doc2vecModel, books)
    users_clust_avg_ratings_with_sums = pd.merge(users_clust_avg_ratings, books, on="isbn", validate="one_to_one")
    vectorized_cluster_books = getVectorizedBooks(users_clust_avg_ratings_with_sums, doc2vecModel, books)
    # Train prediction model
    import doc2vec_networks
    model = doc2vec_networks.trainClusterNetwork(vectorized_cluster_books)

    vect_sums_list = vectorized_rel_unrated_books["Vectorized_Summary"].to_list()
    X = pd.DataFrame(vect_sums_list, columns=["vect"+str(i) for i in range(len(vect_sums_list[0]))])
    predictions = model.predict(X)
    vectorized_rel_unrated_books["NN_Rating"] = predictions
    print(vectorized_rel_unrated_books.head())

#run()