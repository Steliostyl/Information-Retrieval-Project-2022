from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
import pandas as pd
from functions import BOOKS

def createIndex(es: Elasticsearch, idx_name: str = "books") -> None:
  """Creates a new book index, deleting
  any pre-existing with the same name."""

  # Define the mappings of the books index.
  mappings = {
    "properties": {
      "isbn": {"type": "text", "analyzer": "keyword"},
      "book_title": {"type": "text", "analyzer": "english"},
      "book_author": {"type": "text", "analyzer": "standard"},
      "year_of_publication": {"type": "integer"},
      "publisher": {"type": "text", "analyzer": "standard"},
      "summary": {"type": "text", "analyzer": "english"},
      "category": {"type": "text", "analyzer": "standard"}
    }
  }

  # Delete pre-existing index with the same name
  #es.indices.delete(index=idx_name)
  # Create a new index named idx_name with the defined mappings
  es.indices.create(index=idx_name, mappings=mappings)

def insertData(es: Elasticsearch) -> str:
  """Parses the data of a specified csv file and
  inserts the data into Elasticsearch. Returns
  the number of entries after insertion."""

  # Parse the CSV dataset
  dataframe = pd.read_csv(BOOKS).dropna().reset_index()

  # Create a list containing the parsed rows from the CSV file
  bulk_data = []
  for i,row in dataframe.iterrows():
      bulk_data.append(
          {
              "_index": "books",
              "_id": i,
              "_source": { 
                "isbn": row["isbn"],
                "book_title": row["book_title"],
                "book_author": row["book_author"],
                "year_of_publication": row["year_of_publication"],
                "publisher": row["publisher"],
                "summary": row["summary"],
                "category": row["category"]
              }
          }
      )
  # Bulk insert the rows into ElasticSearch
  bulk(es, bulk_data)

  # Refresh the books index and return the number of items in it.
  es.indices.refresh(index="books")
  resp = es.cat.count(index="books", format="json")
  return resp

def makeQuery(es: Elasticsearch, search_string: str) -> tuple:
  """Creates a query and returns Elasticsearch's answer."""

  # Create query body using inputs. Using multi_match we check multiple fields
  # of an entry for the given search string and the final score is calculated
  # by adding the (weighted) score of the fields.
  query_body = {
    "query": {
        "multi_match": {
            "query": search_string,
            "type": "most_fields",
            "fields": ["book_title^1.5", "summary"]
        }
    }
  }

  # Make the query to ElasticSearch
  es_reply = es.search(
    index = "books",
    body = query_body,
    size = 10_000
  )

  return es_reply["hits"]
