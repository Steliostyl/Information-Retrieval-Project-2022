import json
from pprint import pprint
from csv import reader, writer
import re
import sys

USERS_CSV_PATH = "Files/BX-Users.csv"
USERS_JSON_PATH = "Files/BX-Users.json"
USERS_BC_CSV_PATH = "Files/BX-Users-BC.csv"

RATINGS_CSV_PATH = "Files/BX-Book-Ratings.csv"
RATINGS_JSON_PATH = "Files/BX-Book-Ratings.json"
USER_R_WEIGHT = 1.1

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

def combineAllScores(es_reply: dict, user_id: int, user_ratings: dict) -> dict:
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

    #print(f"Best score: {sorted_list[0]}")

    return sorted_list

## Might be useful later
## Check whether isbn is 10 characters and if it's not,
## add 0s in the beginning of the string
#diff = 10-len(isbn)
#isbn = "0" * diff + isbn

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
        csv_writer.writerow(["Country", "ID", "Age"])
        for country in users_by_country:
            for uid in users_by_country[country]:
                csv_writer.writerow([country, uid, users_by_country[country][uid]])

    return users_by_country

def checkForBadEntries(users_b_c: dict) -> None:
    """Prints potential bad entries statistics."""

    bad_countries = 0
    good_countries = 0
    for country in users_b_c.items():
        country_pop = len(country[1])
        if country_pop <= 10:
            bad_countries += 1
            print(country[0], country_pop, country[1])
        else:
            good_countries += 1
    print(bad_countries, good_countries)

createUsersByCountryCSV()