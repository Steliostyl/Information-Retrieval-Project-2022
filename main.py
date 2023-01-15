from pprint import pprint
import es_functions
import functions
from elasticsearch import Elasticsearch

def main():
    # Connect to the ElasticSearch cluster
    es = Elasticsearch("http://localhost:9200")

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

if __name__ == "__main__":
    main()