from pprint import pprint
import es_functions
from elasticsearch import Elasticsearch

def main():
    # Connect to the ElasticSearch cluster
    es = Elasticsearch("http://localhost:9200")

    # Create a new index in ElasticSearch called books
    es_functions.createIndex(es, idx_name="books")

    # Insert a sample of data from dataset into ElasticSearch
    book_count = es_functions.insertData(es, filename="Files/BX-Books.csv")
    print(book_count)

    # Make a query to ElasticSearch and print the results
    hits = es_functions.makeQuery(es)
    pprint(hits)

if __name__ == "__main__":
    main()