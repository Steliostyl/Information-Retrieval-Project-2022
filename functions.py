import re
import pandas as pd
from keras.models import Sequential
from clustering import kPrototypes, kMeans
import timeit
from random import randint
from emb_layer_networks import prepareSummaries
pd.options.mode.chained_assignment = None  # default='warn'

# Paths
INPUT_FILES = "Files/Input/"

USERS = INPUT_FILES + "BX-Users.csv"
RATINGS = INPUT_FILES + "BX-Book-Ratings.csv"
BOOKS = INPUT_FILES + "BX-Books.csv"

PROCESSED_FILES = "Files/Processed/"
CLUSTER_ASSIGNED_USERS = PROCESSED_FILES + "Cluster-Assigned-Users.csv"
AVG_CLUSTER_RATINGS = PROCESSED_FILES + "Average-Cluster-Rating.csv"
CLUSTER_ASSIGNED_USERS = PROCESSED_FILES + "Cluster-Assigned-Users.csv"
PROC_USERS = PROCESSED_FILES + "Processed-Users.csv"

OUTPUT_FILES = "Files/Output/"

SCORES_NO_CLUST = OUTPUT_FILES + "Scores-No-Clustering.csv"
SCORES_W_CLUST = OUTPUT_FILES + "Scores-With-Clustering.csv"

USER_R_WEIGHT = 1.25

# Maximum valid age. Higher ages are considered false entries.
MAX_AGE = 120
# Minimum population of a (probably) valid country.
MIN_VAL_POP = 2
# Rating to use when rating is not found (out of 10)
DEFAULT_RATING = 5
    
def preLoadProcUsers() -> pd.DataFrame:
    while True:
        pre_load = input("Load processed users from file? (y/n): ")
        if pre_load == "y":
            try:
                proc_users = pd.read_csv(PROC_USERS)
                print("Loaded processed users.")
                return proc_users
            except:
                print("Failed to load processed users.")
                break
        elif pre_load == "n":
            break
    print("Processing users...")
    proc_users = processUsersCSV()
    proc_users.to_csv(PROC_USERS,index=False)
    return proc_users

def preLoadClusterAssignement(k: int, proc_users: pd.DataFrame, kM: bool = False) -> pd.DataFrame:
    while True:
        pre_load = input("Load clusters from file? (y/n): ")
        if pre_load == "y":
            try:
                cluster_assigned_users = pd.read_csv(CLUSTER_ASSIGNED_USERS)
                print("Loaded cluster assigned users.")
                return cluster_assigned_users
            except:
                print("Failed to load cluster assigned users.")
                break
        elif pre_load == "n":
            break
    print(f"Clustering users in {k} clusters...")
    if kM:
        cluster_assigned_users = kMeans(k, proc_users)
    else:
        cluster_assigned_users = kPrototypes(k, proc_users)
    cluster_assigned_users.to_csv(CLUSTER_ASSIGNED_USERS, index=False)
    return cluster_assigned_users

def combinedScoreFunc(norm_es_score: float, user_rating: float) -> float:
    """Function that accepts as inputs the elastic search score of a book,
    as well as the user's rating of the book and returns a combined score."""

    # Add the normalized and weighted scores and scale them in the range [0, 10]
    return (USER_R_WEIGHT * user_rating + norm_es_score) / (1 + USER_R_WEIGHT)

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

def createAvgClusterRatings(
        cluster_assignement
    ) -> pd.DataFrame:
    """Function that accepts as input the cluster assignement DataFrame and
    restores User IDs to it. Then, it combines this DF with the Book-Ratings
    CSV, averaging out the book ratings per cluster. The combined DF contains
    the columns isbn, cluster and rating and is finally returned."""

    # Open book ratings CSV in read mode
    books_ratings_df = pd.read_csv(RATINGS)
    # Merge the two DataFrames on UIDs
    result = pd.merge(right=books_ratings_df, left=cluster_assignement,\
        how="left", left_on="User_ID", right_on="uid", validate="one_to_many")
    # Drop useless columns
    result.drop(["uid", "User_ID", "Country", "Country_ID", "Age"], axis=1, inplace=True)
    # Group ratings by isbn and Cluster and sort resulting DataFrame
    avg_cluster_ratings = result.groupby(["isbn", "cluster"], as_index=False).mean()
    #cluster_rating
    avg_cluster_ratings.rename(columns={"rating": "cluster_rating"}, inplace=True)
    avg_cluster_ratings.to_csv(AVG_CLUSTER_RATINGS, index=False)

    return avg_cluster_ratings

def askForPreload(k, processed_users) -> pd.DataFrame:
    while True:
        pre_load = input("Try loading clustered users from file? (y/n): ")
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

def calculateCombinedScoresv2(rel_user_ratings: pd.DataFrame, 
                              rel_unrated_books: pd.DataFrame,
                              rel_clust_rated_books: pd.DataFrame = pd.DataFrame()
                              ) -> pd.DataFrame:
                              
    """Function that accepts as inputs the reply from ElasticSearch, user's id
    and all of their ratings and returns a sorted list of documents with the
    updated combined scores, which are calculated using the combinedScoreFunc."""

    #print(f"Number of relevant user ratings: {len(rel_user_ratings)}")
    #print(f"Number of relevant cluster ratings: {len(rel_clust_rated_books)}")
    #print(f"Number of relevant books not rated by cluster: {len(rel_unrated_books)}")

    if len(rel_user_ratings) > 0:
        #if "score" in rel_user_ratings.columns:
        #    rel_user_ratings.drop("score", axis=1, inplace=True)
        rel_user_ratings["score"] = [combinedScoreFunc(book[1]["norm_es_score"], book[1]["user_rating"]) for book in rel_user_ratings.iterrows()]
    
    if len(rel_clust_rated_books) > 0:
        rel_clust_rated_books["score"] = [combinedScoreFunc(book[1]["norm_es_score"], book[1]["cluster_rating"]) for book in rel_clust_rated_books.iterrows()]
        
    if len(rel_unrated_books) > 0:
        #if "score" in rel_unrated_books.columns:
        #    rel_unrated_books.drop("score", axis=1, inplace=True)
        if "NN_Rating" in rel_unrated_books.columns:
            # Clamp predictions in [0, 10]
            rel_unrated_books["score"] = [combinedScoreFunc(book[1]["norm_es_score"], max(min(10, book[1]["NN_Rating"]), 0)) for book in rel_unrated_books.iterrows()]
        else:
            rel_unrated_books["score"] = [combinedScoreFunc(book[1]["norm_es_score"], DEFAULT_RATING) for book in rel_unrated_books.iterrows()]

    combined_scores = pd.concat([rel_user_ratings, rel_clust_rated_books, rel_unrated_books], ignore_index=True).drop(["uid", "_merge"], axis=1).sort_values(by="score", ascending=False)
            
    # Sort the documents and only keep the best 10%
    combined_scores.sort_values(by="score", ascending=False).head(len(combined_scores)//10)

    # Rearrange columns of df
    reorg_cols = ["score", "norm_es_score", "user_rating", "isbn", "book_title"]
                  #"book_author", "year_of_publication", "publisher", "summary", "category"]

    if "cluster" in combined_scores.columns:
        combined_scores.drop("cluster", axis=1, inplace=True)
        reorg_cols = reorg_cols[:3] + ["cluster_rating"] + reorg_cols[3:]
        if "NN_Rating" in combined_scores.columns:
            reorg_cols = reorg_cols[:4] + ["NN_Rating"] + reorg_cols[4:]

    combined_scores = combined_scores[reorg_cols]

    return combined_scores

def getPredictedRatings(rel_unrated_books: pd.DataFrame,
                        model: Sequential, vocab_size, max_length)\
                            -> tuple[pd.DataFrame, pd.DataFrame]:
    
    # Get books in es_books but not in users_clust_avg_ratings
    summaries = rel_unrated_books["summary"].to_list()
    # Prepare their summaries for nn
    X = prepareSummaries(summaries, vocab_size, max_length)
    # Predict book values
    predictions = model.predict(X)
    rel_unrated_books["NN_Rating"] = predictions

    return rel_unrated_books