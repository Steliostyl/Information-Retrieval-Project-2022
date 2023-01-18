from pprint import pprint
import es_functions
import functions
from elasticsearch import Elasticsearch
import clustering

ES_PASSWORD = "-wq4MlKQgAgo+fnKLy=U"
CERT_FINGERPRINT = "dd:4d:bc:9e:34:74:70:c5:8c:c9:40:b6:eb:d8:c7:89:07:dd:2e:fa:ff:a2:f7:62:aa:52:79:10:6c:60:7a:9a"

def main():
    ############################ ELASTICSEARCH #############################
    
    # Connect to the ElasticSearch cluster
    es = Elasticsearch(
        "https://localhost:9200",
        ssl_assert_fingerprint=CERT_FINGERPRINT,
        basic_auth=("elastic", ES_PASSWORD)
        )

    # Create a new index in ElasticSearch called books
    #es_functions.createIndex(es, idx_name="books")

    # Insert a sample of data from dataset into ElasticSearch
    #book_count = es_functions.insertData(es)
    #print(book_count)

    # Read the user ratings JSON previously created.
    # If the file is not found, create it.
    user_ratings = functions.readUserRatings()

    # Make a query to ElasticSearch and print the results
    es_reply, user_id = es_functions.makeQuery(es)
    #print(len(es_reply["hits"]["hits"]))
    #pprint(es_reply["hits"]["hits"][:5])

    # Get the combined scores for all replies
    combined_scores = functions.combineAllScores(es_reply, user_id, user_ratings)
    # Only keep the best 10% documents
    final_answer = combined_scores[:len(combined_scores)//10]

    # Print the number of documents returned by ElasticSearch,
    # as well as the number of documents in the final reply (10%)
    print(len(combined_scores))
    print(len(final_answer))

    # Print the scores of the 5 best and the 5 worst matches
    pprint([a["_score"] for a in final_answer[:5]])
    pprint([a["_score"] for a in final_answer[-5:]])

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
    combined_data_df = functions.combineUsersCSVs(clustered_data_df)


if __name__ == "__main__":
    main()