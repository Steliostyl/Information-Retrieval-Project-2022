import json
from pprint import pprint
from csv import reader, writer
import re
import sys
import pandas as pd

FILES_PATH = "Files/"

USERS_CSV_PATH = FILES_PATH + "BX-Users.csv"
USERS_JSON_PATH = FILES_PATH + "BX-Users.json"
USERS_BC_CSV_PATH = FILES_PATH + "BX-Users-BC.csv"

RATINGS_CSV_PATH = FILES_PATH + "BX-Book-Ratings.csv"
RATINGS_JSON_PATH = FILES_PATH + "BX-Book-Ratings.json"
USER_R_WEIGHT = 1.1

# Maximum valid age. Higher ages are considered false entries.
MAX_AGE = 120
# Minimum population of a (probably) valid country.
MIN_VAL_POP = 2

def createUserRatingJSON() -> dict:
    """Reads the in_file (CSV) containing user ratings and creates a python dictionary
    whose keys are the user IDs and its values are the corresponding user's ratings."""

    # Create empty user ratings dictionary
    user_ratings = {}

    # Open ratings CSV in read mode
    with open(RATINGS_CSV_PATH, 'r') as input_file:
        csv_reader = reader(input_file)
        # Skip first line (headers)
        _ = next(csv_reader)

        # Iterate the lines of the csv
        for uid, isbn, rating in csv_reader:
            # If uid isn't already a key in user_ratings
            if uid not in user_ratings:
                # Create a dictionary containing the new rating
                # and set it as value of user_ratings[uid]
                user_ratings[uid] = {isbn: rating}
            else:
                # If uid is already a key in user_ratings, add
                # the new rating to its dictionary
                user_ratings[uid][isbn] = rating
        
    # Save ratings dict to JSON file
    with open(RATINGS_JSON_PATH, "w") as output_file:
        json.dump(user_ratings, output_file, indent=2)

    return user_ratings

def readUserRatings() -> dict:
    """Read user ratings from previously created JSON file.
    If it can't be found, create it. Finally, load it to a
    dictionary and return it."""

    try:
        with open(RATINGS_JSON_PATH, "r") as input_file:
            return json.load(input_file)
    except:
        print("User ratings file not found. Creating it...")
        return createUserRatingJSON()

def combinedScoreFunc(es_score: float, max_score: float, user_rating: float) -> float:
    """Function that accepts as inputs the elastic search score of a book, the max score,
    as well as the user's rating of the book and returns a combined score."""

    # Add the normalized and weighted scores
    combined_score = ((es_score / max_score) + (USER_R_WEIGHT * user_rating / 10.0)) * 4
    #print(f"\nUser Rating: {user_rating}\nES Score: {es_score}\nCombined score: {combined_score}")
    return combined_score

def normalizeScores(not_normalized_list: list, max_score: float, normalized_list: list = None) -> list:
    """Normalize scores in not_normalized_list. If a normalized_list is not None,
    final list that is returned is consisted of the normalized_list followed by the
    (now normalized) not_normalized_list."""

    for item in not_normalized_list:
        item["_score"] *= 4 / max_score

    if normalized_list is None:
        return not_normalized_list

    return normalized_list + not_normalized_list

def insertInSortedList(s_list: list, ins_hit: dict) -> list:
    """Inserts ins_hit into sorted list s_list."""

    # If sorted list is empty, return a list consisted only of ins_hit
    if not s_list:
        return [ins_hit]

    # Initialize insertion index to the final element of the sorted list
    insertion_idx = len(s_list) - 1
    while insertion_idx >= 0 and ins_hit["_score"] > s_list[insertion_idx]["_score"]:
        insertion_idx -= 1

    # Return the sorted list with the new element added to it
    return s_list[:insertion_idx+1] + [ins_hit] + s_list[insertion_idx+1:]

def combineAllScores(es_reply: dict, user_id: int, user_ratings: dict, use_cluster_ratings: bool = False,\
    avg_clust_ratings: pd.DataFrame = None, cluster_assigned_users_df: pd.DataFrame = None) -> list:
    """Function that accepts as inputs the reply from ElasticSearch, user's id
    and all of their ratings and returns a sorted list of documents with the
    updated combined scores, which are calculated using the combinedScoreFunc."""

    hits = es_reply["hits"]["hits"]
    max_score = es_reply["hits"]["max_score"]

    # User has not rated any books yet
    if str(user_id) not in user_ratings:
        print(f"No ratings found for user {user_id}.")
        return normalizeScores(hits, max_score)

    # Create a copy of user's ratings
    user_ratings_copy = dict(user_ratings[str(user_id)])
    #pprint(user_ratings_copy)

    # Initialized a new list where elements will
    # be inserted into in a sorted manner
    sorted_list = []

    if not use_cluster_ratings:
        # Iterate through the documents of the ES reply
        for idx, hit in enumerate(hits):
            isbn = hit["_source"]["isbn"]

            # User ratings is empty
            if not user_ratings_copy:
                return normalizeScores(hits, max_score, hits[idx:])

            # User hasn't rated this book
            elif isbn not in user_ratings_copy:
                # Combined score will be the normalized ES score
                hit["_score"] *= 4 / max_score
                # Since the reply from ES is already sorted, only rated
                # books need to be sorted into the list. For unrated
                # books, we just append them to the end of the list. 
                sorted_list.append(hit)

            # User has rated this book
            else:
                # Combined score will be calculated using combinedScoreFunc
                hit["_score"] = combinedScoreFunc(hit["_score"] ,max_score, float(user_ratings_copy[isbn]))
                # Insert document into the sorted list after calculating its new score
                sorted_list = insertInSortedList(s_list=sorted_list, ins_hit=hit)
                # Delete document for user_ratings_copy (in order to know when it's empty)
                del(user_ratings_copy[isbn])
    else:
        # Iterate through the documents of the ES reply
        for idx, hit in enumerate(hits):
            isbn = hit["_source"]["isbn"]
            
            # User ratings is empty or user hasn't rated this book
            if not user_ratings_copy or (isbn not in user_ratings_copy):
                user_rating = getAvgClusterRating(user_id, isbn, avg_clust_ratings, cluster_assigned_users_df)
            else:
                user_rating = float(user_ratings_copy[isbn])


            hit["_score"] = combinedScoreFunc(hit["_score"] ,max_score, user_rating)
            sorted_list = insertInSortedList(s_list=sorted_list, ins_hit=hit)

    #print(f"Best score: {sorted_list[0]}")

    return sorted_list

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
        min_age = 100
        max_age = 0

        # Iterate the lines of the csv
        for uid, location, age in csv_reader:
            try: 
                age = int(float(age))
                if age > MAX_AGE:
                    age = -1
            except: 
                age = -1

            if age != -1:
                if float(age) < min_age:
                    min_age = age
                if age > max_age:
                    max_age = age
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

        mean_age = (min_age + max_age) // 2

        #print(f"Min Age: {min_age} Max Age: {max_age} Mean Age: {mean_age}")
        
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
                    age = mean_age
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
    try:
        users_cluster = cluster_assigned_users_df["Cluster"].loc[cluster_assigned_users_df["User_ID"] == user_id].iloc[0]
        try:
            cluster_avg_rating = avg_clust_ratings.loc[(avg_clust_ratings["isbn"] == isbn) & (avg_clust_ratings["Cluster"] == users_cluster)]
        # If a book hasn't been rated by a cluster, return the median value of 5 stars out of 10
        except:
            cluster_avg_rating = 5
    except:
        return 5

    return cluster_avg_rating