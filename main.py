from elasticsearch import Elasticsearch
from pprint import pprint
import elastic
import functions
#import clustering
import doc2vec
import doc2vec_networks
import emb_layer_networks
import pandas as pd

BOOKS = "Files/Input/BX-Books.csv"

PROCESSED_FILES = "Files/Processed/"

ES_REPLY = PROCESSED_FILES + "ES-Reply.json"
PROC_USERS = PROCESSED_FILES + "Processed-Users.csv"
CLUSTER_ASSIGNED_USERS = PROCESSED_FILES + "Cluster-Assigned-Users.csv"
AVG_CLUSTER_RATINGS = PROCESSED_FILES + "Average-Cluster-Rating.csv"
BOOKS_VECTORIZED_SUMMARIES = PROCESSED_FILES + "Books-Vectorized-Summaries.csv"
BOOKS_VECTORIZED_SUMMARIES_PKL = PROCESSED_FILES + "Books-Vectorized-Summaries.pkl"
D2V_MODEL = PROCESSED_FILES + "Doc2VecModel.mod"
ELASTIC_BOOKS = PROCESSED_FILES + "Elastic_Books.csv"

OUTPUT_FILES = "Files/Output/"

SCORES_NO_CLUST = OUTPUT_FILES + "Scores-No-Clustering.csv"
SCORES_W_CLUST = OUTPUT_FILES + "Scores-With-Clustering.csv"
SCORES_W_CLUST_AND_NN_D2V = OUTPUT_FILES + "Scores-With-Clustering-And-NN-Doc2Vec.csv"
SCORES_W_CLUST_AND_NN_EMB = OUTPUT_FILES + "Scores-With-Clustering-And-NN-Emb-Layer.csv"

ES_PASSWORD = "-wq4MlKQgAgo+fnKLy=U"
CERT_FINGERPRINT = "dd:4d:bc:9e:34:74:70:c5:8c:c9:40:b6:eb:d8:c7:89:07:dd:2e:fa:ff:a2:f7:62:aa:52:79:10:6c:60:7a:9a"

def _input(message, input_type=str):
    """Helper function that forces user
    to enter a predefined type of input."""

    while True:
        try:
            return input_type (input(message))
        except: pass

def useEmbeddingLayer(books: pd.DataFrame, avg_clust_ratings: pd.DataFrame, vs:int, ml: int):
    print(f"Calculating vocab size...")
    print(f"Vocab size: {vs}, Max length: {ml}")
    #model = trainSingleNetwork(books, book_ratings, vocab_size=vs, max_length=ml, classifier=False)
    model = emb_layer_networks.trainClusterNetwork(1, avg_clust_ratings, books, vocab_size=vs, max_length=ml)
    return model

def useDoc2VecModel(books: pd.DataFrame, avg_clust_ratings: pd.DataFrame, users_cluster: int):
    vectorized_books = askForLoadVectorizedBooks(books)
    # Train a neural network model to predict user's cluster average book ratings
    print(f"Trainning a neural network model to predict missing ratings of cluster {users_cluster}...")
    model = doc2vec_networks.trainClusterNetwork(users_cluster, vectorized_books, avg_clust_ratings)
    return vectorized_books, model

def askForLoadVectorizedBooks(books:pd.DataFrame) -> pd.DataFrame:
    while True:
        pre_load = input("Load vectorized books from file? (y/n): ")
        if pre_load == "y":
            try:
                vectorized_books = pd.read_pickle(BOOKS_VECTORIZED_SUMMARIES_PKL)
                print("Loaded vectorized books from file.")
                return vectorized_books
            except:
                print("Couldn't find vectorized books file. Vectorizing summaries...")
                break
        elif pre_load == "n":
            print("Clustering users...")
            break
    vectorized_books = doc2vec.vectorizeSummaries(books_to_vect=books, trainning_books=books)
    print("Saving vectorized summaries...")
    vectorized_books.to_pickle(BOOKS_VECTORIZED_SUMMARIES_PKL)
    print("Finished vectorizing summaries.")
    return vectorized_books

def main():
    ############################ ELASTICSEARCH #############################
    
    # Connect to the ElasticSearch cluster
    es = Elasticsearch(
        "https://localhost:9200",
        ssl_assert_fingerprint=CERT_FINGERPRINT,
        basic_auth=("elastic", ES_PASSWORD)
        )

    # Create a new index in ElasticSearch called books
    #print("Creating books index...")
    #elastic.createIndex(es, idx_name="books")

    # Insert a sample of data from dataset into ElasticSearch
    #book_count = elastic.insertData(es)[0]["count"]
    #print(f"Inserted {book_count} books into index.")

    # Input query parameters and make query to ES
    search_string = input("Enter search string:")
    user_id = _input("Enter your user ID (must be an integer): ", int)
    es_reply = elastic.makeQuery(es, search_string)
    es_books_list = [[entry["_score"]] + (list(entry["_source"].values())) for entry in es_reply["hits"]]
    elastic_books = pd.DataFrame(es_books_list, columns=["score","isbn","book_title","book_author","year_of_publication","publisher","summary","category"])
    elastic_books.to_csv(ELASTIC_BOOKS, index=False)

    # Print results summary
    print(f"\nElasticsearch returned {len(es_reply['hits'])} books. The 5 best matches are:")
    print(elastic_books.head())

    # Get the combined scores for all replies
    print("\nCalculating combined scores...")
    combined_scores, _ = functions.calculateCombinedScores(es_reply, user_id)
    combined_scores.to_csv(SCORES_NO_CLUST, index=False)

    # Print the scores of the 5 best matches
    print("\nBest 5 matches without clustering:\n")
    print(combined_scores.head())

    input("\nPress enter to continue to Clustering...\n")

    ############################## CLUSTERING ##############################

    # Create a CSV containing users sorted by their country
    # Entries that have a high chance of being fake are ignored
    print("Creating processed users CSV...")
    processed_users = functions.processUsersCSV()
    processed_users.to_csv(PROC_USERS, index=False)
    print(processed_users.head(5))

    # Plot elbow curve to help determine optimal number of clusters
    #print("\nPlotting elbow curve...")
    #clustering.plot_elbow_curve(2, 12, 10_000, processed_users)
    k = 3#_input("Choose number of clusters to use: ", int)

    # Generate or load cluster assignement from file and calculate average cluster ratings for their rated books.
    cluster_assignement, avg_clust_ratings = functions.createAvgClusterRatings(k, processed_users)
    cluster_assignement.to_csv(CLUSTER_ASSIGNED_USERS, index=False)
    avg_clust_ratings.to_csv(AVG_CLUSTER_RATINGS, index=False)
    
    print("Re-calculating combined scores using user's cluster's average book ratings...")
    combined_scores_clusters, users_cluster = functions.calculateCombinedScores(
        es_reply, user_id, use_cluster_ratings=True, avg_clust_ratings = avg_clust_ratings,
        cluster_assigned_users = cluster_assignement)
    combined_scores_clusters.to_csv(SCORES_W_CLUST, index=False)
    
    input("\nPress enter to continue to neural networks...\n")

    ############################## NEURAL NETWORK ##############################
    
    books = pd.read_csv(BOOKS)

    # Use a single model with an embedding layer that vectorizes summaries and later predicts missing ratings
    vs, ml = emb_layer_networks.calculateVocab(books)
    model = useEmbeddingLayer(books, avg_clust_ratings, vs, ml)
    combined_scores_clusters_nn, _ = functions.calculateCombinedScores(es_reply, user_id,
        use_cluster_ratings=True, avg_clust_ratings = avg_clust_ratings,
        cluster_assigned_users = cluster_assignement, use_nn=2,
        model=model, books=books, vocab_size=vs, max_length=ml)
    combined_scores_clusters_nn.to_csv(SCORES_W_CLUST_AND_NN_EMB, index=False)  

    ## Use a Doc2Vec model to turn summaries into vectors and then train and use another model to predict the missing ratings
    #vectorized_books, model = useDoc2VecModel(books, avg_clust_ratings, users_cluster)
    #print("Re-calculating combined scores using user's cluster's average book ratings and neural network...")
    #combined_scores_clusters_nn, _ = functions.calculateCombinedScores(es_reply, user_id,
    #    use_cluster_ratings=True, avg_clust_ratings = avg_clust_ratings,
    #    cluster_assigned_users = cluster_assignement, use_nn=1,
    #    model=model, vectorized_books=vectorized_books)
    #combined_scores_clusters_nn.to_csv(SCORES_W_CLUST_AND_NN_D2V, index=False)

if __name__ == "__main__":
    main()