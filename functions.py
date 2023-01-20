from pprint import pprint
from csv import reader, writer
import re
import sys
import pandas as pd

FILES_PATH = "Files/"

USERS_CSV_PATH = FILES_PATH + "BX-Users.csv"
USERS_BC_CSV_PATH = FILES_PATH + "BX-Users-BC.csv"

RATINGS_CSV_PATH = FILES_PATH + "BX-Book-Ratings.csv"

USER_R_WEIGHT = 1.1
ES_WEIGHT = 1

# Maximum valid age. Higher ages are considered false entries.
MAX_AGE = 120
# Minimum population of a (probably) valid country.
MIN_VAL_POP = 2

def combinedScoreFunc(es_score: float, max_score: float, user_rating: float) -> float:
    """Function that accepts as inputs the elastic search score of a book, the max score,
    as well as the user's rating of the book and returns a combined score."""

    # Add the normalized and weighted scores and scale them in the range [-10, 10]
    combined_score = ((ES_WEIGHT * es_score / max_score) + (USER_R_WEIGHT * (user_rating-5) / 5)) * 10 / (USER_R_WEIGHT + ES_WEIGHT)
    #print(f"\nUser Rating: {user_rating}\nES Score: {es_score}\nCombined score: {combined_score}")
    return combined_score

def combineAllScores(es_reply: dict, user_id: int, use_cluster_ratings: bool = False,\
    avg_clust_ratings: pd.DataFrame = None, cluster_assigned_users_df: pd.DataFrame = None) -> pd.DataFrame:
    """Function that accepts as inputs the reply from ElasticSearch, user's id
    and all of their ratings and returns a sorted list of documents with the
    updated combined scores, which are calculated using the combinedScoreFunc."""

    hits = es_reply["hits"]["hits"]
    max_score = es_reply["hits"]["max_score"]
    user_ratings_df = getUserRatings(user_id)

    # List to be filled with book entries
    # and their combined scores
    books_list = []

    # Iterate through the documents of the ES reply
    for hit in hits:
        isbn = hit["_source"]["isbn"]
        
        # User has rated book
        try:
            # Get user's book rating and typecast it to float (from string)
            book_rating = float(user_ratings_df.loc[user_ratings_df["isbn"] == isbn]["rating"].iloc[0])
        # User has not rated this book
        except:
            if not use_cluster_ratings:
                book_rating = 5
            else:
                book_rating = getAvgClusterRating(user_id, isbn, avg_clust_ratings, cluster_assigned_users_df)
    
        # Combined score will be calculated using combinedScoreFunc
        score = combinedScoreFunc(hit["_score"], max_score, book_rating)
        # Create a new book entry as a list
        new_book = [score, isbn, hit['_source']["book_title"], hit['_source']["book_author"], hit['_source']["year_of_publication"], hit['_source']["publisher"], hit['_source']["summary"], hit['_source']["category"]]
        books_list.append(new_book)

    # Create a new dataframe from books_list and sort it by score
    best_matches = pd.DataFrame(data=books_list, columns=["score", "isbn", "book_title", "book_author" , "year_of_publication", "publisher", "summary", "category"])\
        .sort_values(by="score", ascending=False)

    # Only keep the best 10% documents
    return best_matches.head(len(best_matches.index)//10)

def createUsersByCountryCSV() -> dict:
    """Reads the in_file (CSV) containing user ratings and creates a python dictionary
    whose keys are the user IDs and its values are the corresponding user's ratings."""

    # Create empty users dictionary
    users_by_country = {}

    # Open ratings CSV in read mode
    with open(USERS_CSV_PATH, 'r') as input_file:
        csv_reader = reader(input_file, sys.stdout, skipinitialspace=True, lineterminator='\n')
        # Skip first line (headers)
        _ = next(csv_reader)

        # Iterate the lines of the csv
        for uid, location, age in csv_reader:
            try: 
                age = int(float(age))
                if age > MAX_AGE:
                    age = -1
            except: 
                age = -1
                
            # Extract country from location string
            country = re.findall(r"[\s\w+]+$", location)
            if not country:
                #print(uid, location)
                country = ""
            else:
                country = country[0][1:]

            # If country isn't already a key in users_by_country
            if country not in users_by_country.keys():
                # Create a dictionary containing the new user
                # and set it as value of users_by_country[country]
                users_by_country[country] = {uid: age}
            else:
                # If country is already a key in users_by_country,
                # add the new user to its dictionary
                users_by_country[country][uid] = age
        
    # Save ratings dict to new CSV file
    with open(USERS_BC_CSV_PATH, "w", newline='', encoding='utf-8') as output_file:
        csv_writer = writer(output_file)

        # Write header
        csv_writer.writerow(["User_ID", "Country", "Age", "Country_ID"])
        for idx, country in enumerate(users_by_country):
            # Ignore entries with (probably) invalid countries
            if len(users_by_country[country]) < MIN_VAL_POP:
                continue
            elif country == "" or country == " ":
                blank_country = True
            else:
                blank_country = False
                country_idx = idx

            for uid in users_by_country[country]:
                # Entries with blank countries need to have different indexes for clustering!
                if blank_country:
                    country_idx = 500 + int(uid)
                age = users_by_country[country][uid]
                # Age has to be a number for clustering. We fill 
                # false/empty ages with the mean age of our dataset.
                if age < 0 or age > MAX_AGE:
                    age = MAX_AGE/2
                csv_writer.writerow([uid, country, age, country_idx])

    return users_by_country

def printUsersDSstats(users_b_c: dict) -> None:
    """Prints entries statistics."""

    probl_countries = 0
    val_countries = 0
    comp_entries = 0
    ok_entries = 0
    total_entries = 0

    for country in users_b_c.items():
        country_pop = len(country[1])
        if country_pop <= 1:
            probl_countries += 1
            #print(country[0], country_pop, country[1])
        else:
            val_countries += 1
            ok_entries += country_pop
            for user in country[1]:
                if country[1][user] != "" and country[1][user] != " " \
                    and float(country[1][user]) > 0 and float(country[1][user]) < 120:
                    comp_entries += 1
        total_entries += country_pop

    # Print statistics
    print("Dataset Statistics")
    print(f"Total entries: {total_entries}")
    print(f"Entries that contain a (probably) valid country: {ok_entries}")
    print(f"Full entries (containing a valid country and age): {comp_entries}")
    print(f"Total countries: {probl_countries + val_countries}")
    print(f"Countries with very small population (mostly invalid): {probl_countries}")
    print(f"Countries with a larger population: {val_countries}")

def assignClustersToUsers(cluster_assignement: pd.DataFrame | None = None) -> pd.DataFrame:
    """Adds cluster assignements to users and saves the combined DF
    to a CSV. Finally, returns the combined DF."""

    if type(cluster_assignement) is not pd.DataFrame:
        cluster_assignement = pd.read_csv("Files/Clustered-Data.csv")

    cluster_assigned_users = pd.read_csv(USERS_BC_CSV_PATH)
    cluster_assigned_users.drop(["Country_ID"], axis=1, inplace=True)
    cluster_assigned_users.insert(3, "Cluster", cluster_assignement["Cluster"])

    cluster_assigned_users.to_csv(FILES_PATH + "Cluster-Assigned-Users.csv", index=False)

    return cluster_assigned_users

def createAvgClusterRatings(cluster_assignement_df: pd.DataFrame) -> pd.DataFrame:
    """Function that accepts as input the cluster assignement DataFrame and restores User IDs to it.
    Then, it combines this DF with the Book-Ratings CSV, averaging out the book ratings per cluster.
    The combined DF contains the columns isbn, cluster and rating and is finally returned."""

    # Open book ratings CSV in read mode
    books_ratings_df = pd.read_csv(RATINGS_CSV_PATH)
    # Merge the two DataFrames on UIDs
    result = pd.merge(right=books_ratings_df, left=cluster_assignement_df,\
        how="left", left_on="User_ID", right_on="uid", validate="one_to_many")
    # Drop useless columns
    result.drop(["uid", "User_ID", "Country", "Age"], axis=1, inplace=True)
    # Group ratings by isbn and Cluster and sort resulting DataFrame
    avg_ratings = result.groupby(["isbn", "Cluster"])
    avg_ratings = avg_ratings.mean().sort_values(by=["Cluster", "isbn"])
    # Save DataFrame to CSV
    avg_ratings.to_csv(FILES_PATH + "Average-Cluster-Ratings.csv")

    return avg_ratings

def getAvgClusterRating(user_id: int, isbn: str, avg_clust_ratings: pd.DataFrame, cluster_assigned_users_df: pd.DataFrame) -> float:
    """Given a user id and an book's isbn, it returns the average rating of user's cluster for the specified book."""

    try:
        # Try getting user's cluster. If it can't be found, return the median value of 5 stars out of 10
        users_cluster = cluster_assigned_users_df["Cluster"].loc[cluster_assigned_users_df["User_ID"] == user_id].iloc[0]
        # Try getting user's cluster's rating of specified book
        try:
            cluster_avg_rating = avg_clust_ratings.loc[(avg_clust_ratings["isbn"] == isbn) & (avg_clust_ratings["Cluster"] == users_cluster)]
        # If a book hasn't been rated by a cluster, return the median value of 5 stars out of 10
        except:
            cluster_avg_rating = 5
    except:
        return 5

    return cluster_avg_rating

def getUserRatings(user_id: int, filename: str = RATINGS_CSV_PATH) -> pd.DataFrame:
    """Read ratings CSV and return specified user's ratings."""

    users_ratings_df = pd.read_csv(filename)
    return  pd.read_csv(filename).loc[users_ratings_df["uid"] == user_id]