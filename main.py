from pprint import pprint
import es_functions
from elasticsearch import Elasticsearch

def main():
    es = Elasticsearch("http://localhost:9200")

    #es_functions.createIndex(es)
    book_count = es_functions.insertData(es)
    print(book_count)

    hits = es_functions.makeQuery(es)
    pprint(hits)

if __name__ == "__main__":
    main()