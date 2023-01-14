import json
from pprint import pprint
from csv import reader

RATINGS_CSV_PATH = "Files/BX-Book-Ratings.csv"
RATINGS_JSON_PATH = "Files/BX-Book-Ratings.json"

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

    print(user_ratings["2"]["0195153448"])

    return user_ratings

def getUserRating(uid: int, isbn: str) -> int | None:
    """Returns user's rating of a specific book."""

    # Check whether isbn is 10 characters and if it's not,
    # add 0s in the beginning of the string
    diff = 10-len(isbn)
    isbn = "0" * diff + isbn

    # Open user ratings JSON
    with open(RATINGS_JSON_PATH) as json_file:
        ratings = json.load(json_file)

    str_uid = str(uid)

    if str_uid not in ratings.keys():
        print(f"User {uid} has not rated any books yet.")
        return
    elif isbn not in ratings[str_uid].keys():
        print(ratings[str_uid])
        print(f"User {uid} has not rated book with isbn {isbn} yet.")
        return
    else:
        stars = ratings[str_uid][isbn]
        print(f"User {uid} has rated book with isbn {isbn} with {stars} stars.")
        return stars
        