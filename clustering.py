import pandas as pd
from sklearn.cluster import KMeans
from kmodes.kprototypes import KPrototypes
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
import seaborn
from os import cpu_count

PROCESSED_FILES = "Files/Processed/"
PROC_USERS = PROCESSED_FILES + "Processed-Users.csv"

# Number of jobs to use for parallelism
PC = 12
# Cetroid initializations
N_INIT = 8

def getClusteringInput(proc_users: pd.DataFrame):
    """Reads Users CSV, loads it to a DataFrame, drops 
    unnecessary columns and standardizes 'Age' column.
    Finally, it returns the processed DataFrame."""

    normalized_age_users = proc_users.drop(["Country", "User_ID"], axis=1)
    scaled_X = StandardScaler().fit_transform(normalized_age_users[['Age']])
    normalized_age_users[['Age']] = scaled_X
    return normalized_age_users.values
    
def plot_elbow_curve(start: int, end: int, sample_size: int, proc_users: pd.DataFrame, kM: bool = False) -> None:
    """Plots elbow curve. Used for optimizing K."""

    if kM == True:
        X = StandardScaler().fit_transform(proc_users[['Age']])[:sample_size]
        y = proc_users["Country_ID"].values[:sample_size]
    else:
        X = getClusteringInput(proc_users)

    categorical_index = [1]
    no_of_clusters = list(range(start, end+1))
    cost_values = []
    threads = min(PC, cpu_count())
    print(f"Starting testing using {threads} threads...")
    
    for k in no_of_clusters:
        print(f"Testing with {k} clusters...")
        if kM:
            test_model = KMeans(k, n_init=N_INIT)
            test_model.fit_predict(X, y)
            cost_values.append(test_model.inertia_)
        else:
            test_model = KPrototypes(
                n_clusters=k, init='Huang', n_init=N_INIT, n_jobs=threads
            )
            test_model.fit_predict(X, categorical=categorical_index)
            cost_values.append(test_model.cost_)
        
    seaborn.set_theme(style="whitegrid", palette="bright", font_scale=1.1)
    
    plt.figure(figsize=(15, 7))
    ax = seaborn.lineplot(
        x=no_of_clusters, y=cost_values, marker="o", dashes=False
    )
    ax.set_title('Elbow curve', fontsize=18)
    ax.set_xlabel('No of clusters', fontsize=14)
    ax.set_ylabel('Cost', fontsize=14)
    ax.set(xlim=(start-0.1, end+0.1))
    plt.plot()
    plt.show()

def kPrototypes(k: int, proc_users: pd.DataFrame) -> pd.DataFrame:
    """Runs k-Prototypes algorithm for Users, placing them
    in one of k clusters."""

    categorical_features_idx = [1]
    mark_array = getClusteringInput(proc_users)
    threads = min(PC, cpu_count())
    print(f"Number of available threads: {cpu_count()}")
    print(f"Starting k-Prototypes using {threads} threads...")
    model = KPrototypes(
        n_clusters=k, verbose=1, init='Huang', n_init=N_INIT, n_jobs=threads
    )
    clusters = model.fit_predict(
        mark_array, categorical=categorical_features_idx
    )
    # Add Cluster column to dataframe
    proc_users['cluster'] = list(clusters)

    return proc_users

def kMeans(k: int, proc_users: pd.DataFrame) -> pd.DataFrame:
    """Runs k-Prototypes algorithm for Users, placing them
    in one of k clusters."""

    X = StandardScaler().fit_transform(proc_users[['Age']])
    Y = proc_users["Country_ID"].values
    print(X.shape)
    print(Y.shape)
    #threads = min(PC, cpu_count())
    #print(f"Number of available threads: {cpu_count()}")
    #print(f"Starting k-Prototypes using {threads} threads...")
    #X = km_in["Age"].to_list()
    #Y = km_in["Country_ID"].to_list()
    model = KMeans(
        n_clusters=k, verbose=1, n_init=N_INIT
    )
    clusters = model.fit_predict(
        X, Y
    )
    # Add Cluster column to dataframe
    proc_users['cluster'] = list(clusters)

    return proc_users

def run():
    proc_users = pd.read_csv(PROC_USERS)
    clust_assignement = kMeans(3, proc_users)
    print(clust_assignement.head(20))

#run()