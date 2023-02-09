from elasticsearch import Elasticsearch
from pprint import pprint
import elastic
import functions
import clustering
import doc2vec
import doc2vec_networks
import emb_layer_networks
import pandas as pd

# Paths
INPUT_FILES = "Files/Input/"

RATINGS = INPUT_FILES + "BX-Book-Ratings.csv"
BOOKS = INPUT_FILES + "BX-Books.csv"

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


def main():
    ############################ ELASTICSEARCH #############################
    
    # Connect to the ElasticSearch cluster
    es = Elasticsearch(
        "http://localhost:9200",
        #ssl_assert_fingerprint=CERT_FINGERPRINT,
        #basic_auth=("elastic", ES_PASSWORD)
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
    max_score = es_reply["max_score"]
    es_books_list = [[10*entry["_score"]/max_score] + (list(entry["_source"].values())) for entry in es_reply["hits"]]
    elastic_books = pd.DataFrame(es_books_list, columns=["norm_es_score","isbn","book_title","book_author","year_of_publication","publisher","summary","category"])
    elastic_books.to_csv(ELASTIC_BOOKS, index=False)

    # Print results summary
    print(f"\nElasticsearch returned {len(es_reply['hits'])} books. The 5 best matches are:")
    print(elastic_books.head(10))

    # Get user ratings
    ratings = pd.read_csv(RATINGS)
    user_ratings = ratings.loc[ratings["uid"] == user_id]
    user_ratings.rename(columns={"rating": "user_rating"}, inplace=True)
    df = pd.merge(elastic_books, user_ratings, on=['isbn'], how="outer", indicator=True)
    rel_user_ratings = df[df["_merge"] == "both"]
    rel_elastic_unrated_books = df[df["_merge"] == "left_only"]


    # Get the combined scores for all replies
    print("\nCalculating combined scores...")
    #combined_scores, _ = functions.calculateCombinedScores(es_reply, user_id)
    combined_scores = functions.calculateCombinedScoresv2(rel_user_ratings, rel_elastic_unrated_books)
    combined_scores.to_csv(SCORES_NO_CLUST, index=False)

    # Print the scores of the 5 best matches
    print("\nBest 5 matches without clustering:\n")
    print(combined_scores.head(10))

    input("\nPress enter to continue to Clustering...\n")

    ############################## CLUSTERING ##############################

    # Create a CSV containing users sorted by their country
    # Entries that have a high chance of being fake are ignored
    print("Creating processed users CSV...")
    processed_users = functions.preLoadProcUsers()

    # Plot elbow curve to help determine optimal number of clusters
    #print("\nPlotting elbow curve...")
    #clustering.plot_elbow_curve(2, 12, 10_000, processed_users)
    k = 3#_input("Choose number of clusters to use: ", int)

    # Generate or load cluster assignement from file and calculate average cluster ratings for their rated books.
    cluster_assignement = functions.preLoadClusterAssignement(k, processed_users)

    print("Calculating average cluster ratings...")
    avg_clust_ratings = functions.createAvgClusterRatings(cluster_assignement)

    # Try getting user's cluster 
    try:
        users_cluster = cluster_assignement["cluster"].\
                        loc[cluster_assignement["User_ID"] == user_id].iloc[0]
        print(f"User {user_id} belongs in cluster {users_cluster}")
        # Get average cluster ratings of user's cluster
        users_clust_avg_ratings = avg_clust_ratings.loc[avg_clust_ratings["cluster"] == users_cluster]
        df = pd.merge(elastic_books, users_clust_avg_ratings, on=['isbn'], how="outer", indicator=True)
        # Get books in es_books but not in users_clust_avg_ratings
        rel_unrated_books = df[df['_merge'] == 'left_only']
        # Get intersection of es_books and users_clust_avg_ratings
        rel_clust_rated_books = df[df['_merge'] == 'both']
    except:
        rel_clust_rated_books = pd.DataFrame()
        users_cluster = -1
        print(f"User {user_id} wasn't found in BX-Users.csv")
    
    print("Re-calculating combined scores using user's cluster's average book ratings...")
    combined_scores_clusters = functions.calculateCombinedScoresv2(rel_user_ratings, rel_unrated_books, rel_clust_rated_books)
    combined_scores_clusters.to_csv(SCORES_W_CLUST, index=False)
    
    print("\nBest 10 matches with clustering:\n")
    print(combined_scores_clusters.head(10))
    
    input("\nPress enter to continue to neural networks...\n")

    ############################## NEURAL NETWORK ##############################
    
    books = pd.read_csv(BOOKS)
    book_ratings = pd.read_csv(RATINGS)

    # Use a single model with an embedding layer that vectorizes
    # summaries and later predicts missing ratings
    print(f"Calculating vocabulary size and max summary length...")
    vocab_size, max_length = emb_layer_networks.calculateVocab(books)
    print(f"Vocab size: {vocab_size}, Max length: {max_length}")
    if len(rel_clust_rated_books) > 0:
        # Train a neural network on user's cluster ratings
        model = emb_layer_networks.trainClusterNetwork(users_clust_avg_ratings,
                                                       books, vocab_size,
                                                       max_length)
    else:
        model = emb_layer_networks.trainSingleNetwork(books, book_ratings,
                                                      vocab_size, max_length)
        
    # Get the predicted ratings of unratted books
    rel_nn_rated_books = functions.getPredictedRatings(rel_unrated_books, model, vocab_size, max_length)

    print("Re-calculating combined scores using user's cluster's average book ratings as well as the nn...")
    combined_scores_clusters_nn = functions.calculateCombinedScoresv2(rel_user_ratings, rel_nn_rated_books, rel_clust_rated_books)
    combined_scores_clusters_nn.to_csv(SCORES_W_CLUST_AND_NN_EMB, index=False)  
    print("\nBest 10 matches with clustering and neural network:\n")
    print(combined_scores_clusters_nn.head(10))


    ## Use a Doc2Vec model to turn summaries into vectors and then train and use another model to predict the missing ratings
    #doc2vecModel = doc2vec.getDoc2VecModel(books)
    #vectorized_rel_unrated_books = doc2vec.getVectorizedBooks(rel_unrated_books, doc2vecModel, books)
    #users_clust_avg_ratings_with_sums = pd.merge(users_clust_avg_ratings, books, on="isbn", validate="one_to_one")
    #vectorized_cluster_books = doc2vec.getVectorizedBooks(users_clust_avg_ratings_with_sums, doc2vecModel, books)
    ## Train prediction model
    #model = doc2vec_networks.trainClusterNetwork(vectorized_cluster_books)
#
    #predictions = model.predict(vectorized_rel_unrated_books)
    #vectorized_rel_unrated_books["NN_Rating"] = predictions
    #combined_scores_clusters_nn_d2v = functions.calculateCombinedScoresv2(rel_user_ratings, vectorized_rel_unrated_books, rel_clust_rated_books) 
    #combined_scores_clusters_nn.to_csv(SCORES_W_CLUST_AND_NN_D2V, index=False)  
    #print("\nBest 10 matches with clustering and doc2vec neural network:\n")
    #print(combined_scores_clusters_nn_d2v.head(10))

if __name__ == "__main__":
    main()