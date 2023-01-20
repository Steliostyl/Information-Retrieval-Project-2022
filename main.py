from elasticsearch import Elasticsearch
from pprint import pprint
import pandas as pd
import es_functions
import functions
import clustering

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
    #es_functions.createIndex(es, idx_name="books")

    # Insert a sample of data from dataset into ElasticSearch
    #book_count = es_functions.insertData(es)[0]["count"]
    #print(f"Inserted {book_count} books into index.")

    # Make a query to ElasticSearch and print the results
    # Input search string and user ID
    search_string = input("Enter search string:")
    user_id = _input("Enter your user ID (must be an integer): ", int)
    es_reply = es_functions.makeQuery(es, search_string, user_id)
    # Print the length of the reply from ElasticSearch,
    # as well as the best 5 matches.
    print(f'ElasticSearch returned {len(es_reply["hits"]["hits"])} documents. The 5 best are:\n')
    for hit in es_reply["hits"]["hits"][:5]:
        print(hit['_score'], hit['_source']['book_title'])

    # Get the combined scores for all replies
    print("\nCalculating combined scores...")
    combined_scores_df = functions.calculateCombinedScores(es_reply, user_id)
    combined_scores_df.to_csv("Files/Scores-No-Clustering.csv", index=False)
    # Print the scores of the 5 best matches
    print("Best 5 matches without clustering:\n")
    print(combined_scores_df.head(5).drop(["isbn","book_author","year_of_publication","publisher","summary","category"], axis=1))

    input("Press any key to continue to Clustering...\n")

    ############################## CLUSTERING ##############################

    # Create a CSV containing users sorted by their country
    # Entries that have a high chance of being problematic are ignored
    print("Creating processed users CSV...")
    users_by_country = functions.createUsersByCountryCSV()
    functions.printUsersDSstats(users_by_country)
    users_bc_df = pd.DataFrame.from_dict(users_by_country)

    # Plot elbow curve to help determine optimal number of clusters
    print("Plotting elbow curve...")
    clustering.plot_elbow_curve(2, 10, sample_size=10_000)
    k = _input("Choose number of clusters to use: ", int)

    # Run k-Prototypes on slightly cleaned dataset
    clustered_data_df = clustering.kPrototypes(k)
    print("\nCreating cluster assignement CSV...")
    # Add cluster assignements to Users-BC and save them to new CSV
    cluster_assigned_users_df = functions.assignClustersToUsers(clustered_data_df, users_bc_df)

    # Create a dataframe containing the average
    # rating of every book per cluster
    print("Creating average cluster ratings CSV...")
    avg_clust_ratings = functions.createAvgClusterRatings(cluster_assigned_users_df)

    # Get the re-calculated document scores by using
    # user's cluster's average ratings to "rate" unrated books
    print("Re-calculating combined scores using user's cluster's average book ratings...")
    combined_scores_clusters_df = functions.calculateCombinedScores(es_reply, user_id,\
        use_cluster_ratings=True, avg_clust_ratings = avg_clust_ratings,\
            cluster_assigned_users_df = cluster_assigned_users_df)
    combined_scores_clusters_df.to_csv("Files/Scores-With-Clustering.csv", index=False)


if __name__ == "__main__":
    main()