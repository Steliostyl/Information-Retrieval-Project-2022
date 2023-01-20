import pandas as pd
import numpy as np
import regex as re
import timeit

FILES_PATH = "Files/"

USERS_CSV_PATH = FILES_PATH + "BX-Users.csv"

MAX_AGE = 120
H = 5

def extractCountry(df: pd.DataFrame) -> str:
    """Uses a regular expression to extract the country from the location string."""

    country = re.findall(r"[\s\w+]+$", df['location'])
    if not country:
        country = ""
    else:
        country = country[0][1:]

    return country

def getCountryID(df: pd.DataFrame, unique_countries: np.array):
    """Calculates country_id by finding each row's country index from unique countries"""

    country = df['country']
    # Entries with blank countries need to have different indexes for clustering!
    if country == "":
        return 500 + df['uid']

    # Return the unique index of the country
    return np.where(unique_countries == country)[0][0]

def createProcessedUsersCSV() -> pd.DataFrame:
    """Processes users CSV, cleans it up and saves it to a new file."""

    users_df = pd.read_csv(USERS_CSV_PATH)
    # Fill NaN values with the median age (60 in our case)
    users_df.age = users_df.age.fillna(MAX_AGE/2)
    # Fill illegal age values with the median age (60 in our case)
    users_df['age'] = np.where((users_df['age'] < 0) | (users_df['age'] > MAX_AGE), MAX_AGE/2, users_df['age'])
    # Extract country from location string and add it to a new column
    users_df['country'] = users_df.apply(lambda row: extractCountry(row), axis=1)
    # Get the unique countries of the DataFrame
    unique_countries = users_df['country'].unique()
    # Calculate country_id
    users_df['country_id'] = users_df.apply(lambda row: getCountryID(row, unique_countries), axis=1)
    # Drop unnecassary columns
    users_df.drop(['location', 'country'], axis=1, inplace=True)
    #print(users_df.head())
    users_df.to_csv("Files/Processed-Users.csv")
    
    return users_df

num_runs = 3
duration = timeit.Timer(createProcessedUsersCSV).timeit(number = num_runs)
avg_duration = duration/num_runs
print(f'On average it took {avg_duration} seconds')