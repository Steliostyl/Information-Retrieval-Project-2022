from pprint import pprint
import es_functions
import functions
from elasticsearch import Elasticsearch
import clustering
import pandas as pd

ES_PASSWORD = "-wq4MlKQgAgo+fnKLy=U"
CERT_FINGERPRINT = "dd:4d:bc:9e:34:74:70:c5:8c:c9:40:b6:eb:d8:c7:89:07:dd:2e:fa:ff:a2:f7:62:aa:52:79:10:6c:60:7a:9a"

def main():
    ############################ ELASTICSEARCH #############################
    
    # Connect to the ElasticSearch cluster
    es = Elasticsearch(
        "http://localhost:9200",
        #ssl_assert_fingerprint=CERT_FINGERPRINT,
        #basic_auth=("elastic", ES_PASSWORD)
        )

    # Create a new index in ElasticSearch called books
    #es_functions.createIndex(es, idx_name="books")

    # Insert a sample of data from dataset into ElasticSearch
    #book_count = es_functions.insertData(es)
    #print(book_count)

    # Make a query to ElasticSearch and print the results
    es_reply, user_id = es_functions.makeQuery(es)
    #print(len(es_reply["hits"]["hits"]))
    #pprint(es_reply["hits"]["hits"][:5])

    # Get the combined scores for all replies
    combined_scores_df = functions.combineAllScores(es_reply, user_id)
    combined_scores_df.to_csv("Files/Scores-No-Clustering.csv", index=False)

    input("test_test_1_2")

    # Print the scores of the 5 best and the 5 worst matches
    pprint(combined_scores_df.head(5))
    pprint(combined_scores_df.head(-5))

    input("Press any key to continue to Clustering...\n")

    ############################## CLUSTERING ##############################

    # Create a CSV containing users sorted by their country
    # Entries that have a high chance of being problematic are ignored
    users_by_country = functions.createUsersByCountryCSV()
    functions.printUsersDSstats(users_by_country)

    # Plot elbow curve to help determine optimal number of clusters
    clustering.plot_elbow_curve(2, 8)
    print("As we can tell from our elbow curve, the optimal number of clusters is 3.\n")
    input("Press any key to continue...")

    # Run k-Prototypes on slightly cleaned dataset
    clustered_data_df = clustering.kPrototypes()
    # Add cluster assignements to Users-BC and save them to new CSV
    cluster_assigned_users_df = functions.assignClustersToUsers(clustered_data_df)

    # Create a dataframe containing the average
    # rating of every book per cluster
    avg_clust_ratings = functions.createAvgClusterRatings(cluster_assigned_users_df)

    # Get the re-calculated document scores by using
    # user's cluster's average ratings to "rate" unrated books
    combined_scores_clusters = functions.combineAllScores(es_reply, user_id,\
        use_cluster_ratings=True, avg_clust_ratings = avg_clust_ratings,\
            cluster_assigned_users_df = cluster_assigned_users_df)
    combined_scores_clusters = combined_scores_clusters[:len(combined_scores_clusters)//10]

    print(combined_scores_clusters[:5])
    combined_scores_clusters_df = pd.DataFrame.from_dict(combined_scores_clusters)
    combined_scores_clusters_df.to_csv("Files/Scores-With-Clustering.csv", index=True)


if __name__ == "__main__":
    main()