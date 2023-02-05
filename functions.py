import re
import pandas as pd
from keras.models import Sequential
from clustering import kPrototypes
import timeit
from random import randint
from emb_layer_networks import getNetworkInput, pad_sequences, preProcessSummaryv3, one_hot

# Paths
INPUT_FILES = "Files/Input/"

USERS = INPUT_FILES + "BX-Users.csv"
RATINGS = INPUT_FILES + "BX-Book-Ratings.csv"
BOOKS = INPUT_FILES + "BX-Books.csv"

PROCESSED_FILES = "Files/Processed/"
CLUSTER_ASSIGNED_USERS = PROCESSED_FILES + "Cluster-Assigned-Users.csv"

USER_R_WEIGHT = 1.25

# Maximum valid age. Higher ages are considered false entries.
MAX_AGE = 120
DEF_AGE = MAX_AGE // 2
# Minimum population of a (probably) valid country.
MIN_VAL_POP = 2
# Rating to use when rating is not found (out of 10)
DEFAULT_RATING = 5

def combinedScoreFunc(norm_es_score: float, user_rating: float) -> float:
    """Function that accepts as inputs the elastic search score of a book,
    as well as the user's rating of the book and returns a combined score."""

    # Add the normalized and weighted scores and scale them in the range [0, 10]
    return (USER_R_WEIGHT * user_rating + norm_es_score) / (1 + USER_R_WEIGHT)

def calculateCombinedScores(es_reply: dict, user_id: int, use_cluster_ratings:
        bool = False, avg_clust_ratings: pd.DataFrame = None,
        cluster_assigned_users: pd.DataFrame = None, use_nn = 0,
        model: Sequential = None, vectorized_books: pd.DataFrame = None,
        books: pd.DataFrame = None, vocab_size: int = None, max_length: int = None) -> pd.DataFrame:
    """Function that accepts as inputs the reply from ElasticSearch, user's id
    and all of their ratings and returns a sorted list of documents with the
    updated combined scores, which are calculated using the combinedScoreFunc."""

    es_books = es_reply["hits"]
    max_score = es_reply["max_score"]
    user_ratings = getUserRatings(user_id)
    print(f"User {user_id} has rated {len(user_ratings)} book(s).")

    # List to be filled with book entries
    # and their combined scores
    books_list = []
    users_cluster = -1

    # Get user's cluster
    if use_cluster_ratings:
        try:
            users_cluster = cluster_assigned_users["Cluster"].\
                loc[cluster_assigned_users["User_ID"] == user_id].iloc[0]
            print(f"User {user_id} belongs in cluster {users_cluster}")
        except:
            print(f"User {user_id} doesn't exist in database.")
            print("Scores with clustering will be the same as scores without clustering.")

    # Iterate through the documents of the ES reply
    for book in es_books:
        isbn = book["_source"]["isbn"]
        norm_es_score = 10*book["_score"]/max_score
        
        # User has rated book
        try:
            # Get user's book rating and typecast it to float (from string)
            user_rating = float(user_ratings.\
                loc[user_ratings["isbn"] == isbn]["rating"].iloc[0])
        # User has not rated this book
        except:
            # User doesn't exist in the database or function has 
            # been called with arg use_cluster_ratings = False
            if users_cluster == -1:
                user_rating = DEFAULT_RATING
            else:
                user_rating = getAvgClusterRating(
                    users_cluster, isbn, avg_clust_ratings,
                    use_nn, model, vectorized_books, books,
                    vocab_size=vocab_size, max_length=max_length
                )
    
        # Combined score will be calculated using combinedScoreFunc
        score = combinedScoreFunc(norm_es_score, user_rating)
        # Create a new book entry as a list
        new_book = [
            score, user_rating, norm_es_score, isbn,
            book['_source']["book_title"],
            book['_source']["book_author"],
            book['_source']["year_of_publication"],
            book['_source']["publisher"],
            book['_source']["summary"],
            book['_source']["category"]
            ]
        books_list.append(new_book)

    # Create a new dataframe from books_list and sort it by score
    best_matches = pd.DataFrame(data=books_list, columns=[
        "score", "user_rating", "es_scaled_score", "isbn", "book_title",
        "book_author", "year_of_publication", "publisher", "summary", 
        "category"]).sort_values(by="score", ascending=False)

    # Only keep the best 10% documents
    return (best_matches.head(len(best_matches.index)//10), users_cluster)

def processUsersCSV() -> pd.DataFrame:
    """Extract vital information from BX-Users csv and save it to a new CSV."""
    # Create empty users dictionary
    users = []
    countries = []

    # Iterate BX-Users.csv rows
    for entry in pd.read_csv(USERS).iterrows():
        # Cast age to int
        uid = int(entry[1][0])
        # Get location
        location = entry[1][1]
        # Try to cast age to int. If it's empty or
        # out of bounds, set it to the default age
        try: 
            age = int(entry[1][2])
            if age > MAX_AGE or age < 0:
                age = randint(0, MAX_AGE)
        except:
                age = randint(0, MAX_AGE)
            
        # Extract country from location string
        country = re.findall(r"[\s\w+]+$", location)
        # If country is empty, set country_id to 1000+uid, so that users with
        # incomplete countries do not belong in the same "country". Essential
        # for clustering! Also, 1000 is arbitrarily chosen butis empirically
        # higher than the count of normal countries, so there are no conflicts.
        if not country:
            country = ""
            country_id = 1000 + uid
        else:
            country = country[0][1:]

        # If country isn't already in the countries list,
        # its index will be equal to the length of countries
        if country not in countries:
            country_id = len(countries)
            countries.append(country)
        # Otherwise, its index is equal to the index of the country in countries
        else:
            country_id = countries.index(country)
        users.append([uid, age, country, country_id])
    
    users_df = pd.DataFrame(
        users, columns=["User_ID", "Age", "Country", "Country_ID"]
    )
    
    return users_df

def padSummary(summary, max_length):
    padded_sum = summary.extend([0]*(max_length - len(summary)))
    print(summary)
    print(padded_sum)
    return padded_sum

def createAvgClusterRatings(
        k, processed_users
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Function that accepts as input the cluster assignement DataFrame and
    restores User IDs to it. Then, it combines this DF with the Book-Ratings
    CSV, averaging out the book ratings per cluster. The combined DF contains
    the columns isbn, cluster and rating and is finally returned."""

    cluster_assignement = askForPreload(k , processed_users)
    # Open book ratings CSV in read mode
    books_ratings_df = pd.read_csv(RATINGS)
    # Merge the two DataFrames on UIDs
    result = pd.merge(right=books_ratings_df, left=cluster_assignement,\
        how="left", left_on="User_ID", right_on="uid", validate="one_to_many")
    # Drop useless columns
    result.drop(["uid", "User_ID", "Country", "Country_ID", "Age"], axis=1, inplace=True)
    # Group ratings by isbn and Cluster and sort resulting DataFrame
    avg_cluster_ratings = result.groupby(["isbn", "Cluster"], as_index=False).mean()

    return (cluster_assignement, avg_cluster_ratings)

def getAvgClusterRating(users_cluster: int, isbn: str, avg_clust_ratings: pd.DataFrame, use_nn, model: Sequential = None,
                        vectorized_books: pd.DataFrame = None, books: pd.DataFrame = None, vocab_size: int = None,
                        max_length: int = None) -> tuple:
    """Given a user id and an book's isbn, it returns the average rating of user's cluster for the specified book."""

    # Try getting user's cluster's rating of specified book
    try:
        rating = avg_clust_ratings.loc[(avg_clust_ratings["isbn"] == isbn) & (avg_clust_ratings["Cluster"] == users_cluster)]["rating"].iloc[0]
        #print(rating)
        return rating
    # If a book hasn't been rated by a cluster, return the median value of 5 stars out of 10
    except:
        # Using vectorized books
        if use_nn == 1:
            vect_sum = vectorized_books["Vectorized_Summary"].loc[vectorized_books["isbn"] == isbn].to_list()[0].reshape(1, -1)
            return model.predict(vect_sum)[0][0]*10
        # Not using vectorized books
        elif use_nn == 2:
            summary = books[books["isbn"]==isbn]["summary"].to_list()[0]
            # Preprocess summary
            preproc_summary = preProcessSummaryv3(summary)
            # One hot encode words of documents
            encoded_sum = one_hot(preproc_summary, vocab_size)
            # Add padding
            X = pad_sequences([encoded_sum],maxlen=max_length,padding='post')
            return model.predict(X)[0][0]*10
        return DEFAULT_RATING

def getUserRatings(user_id: int, filename: str = RATINGS) -> pd.DataFrame:
    """Read ratings CSV and return specified user's ratings."""

    users_ratings_df = pd.read_csv(filename)
    return  users_ratings_df.loc[users_ratings_df["uid"] == user_id]

def askForPreload(k, processed_users) -> pd.DataFrame:
    while True:
        pre_load = input("Try loading pre-trained clustered users from file? (y/n): ")
        if pre_load == "y":
            try:
                clust_assigned_users = pd.read_csv(CLUSTER_ASSIGNED_USERS)
                print("Loaded clustered users from file.")
                return clust_assigned_users
            except:
                print("Model couldn't be loaded.\Clustering users...")
                break
        elif pre_load == "n":
            print("Clustering users...")
            break
    
    start_time = timeit.default_timer()
    cluster_assigned_users = kPrototypes(k, processed_users)
    elapsed_time = timeit.default_timer() - start_time
    print(f"Clustering with {k} clusters took {int(elapsed_time//60)} minutes and {int(round(elapsed_time % 60, 0))} seconds.")
    cluster_assigned_users.to_csv(CLUSTER_ASSIGNED_USERS, index=False)
    return cluster_assigned_users