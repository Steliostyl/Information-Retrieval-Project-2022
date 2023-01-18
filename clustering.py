import numpy as np
import pandas as pd
from kmodes.kprototypes import KPrototypes
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
import seaborn
from os import cpu_count

# Number of jobs to use for parallelism
PC = 10
# Number of clusters
K = 3

def getUsersDF(filename: str = "Files/BX-Users-BC.csv"):
    """Reads Users CSV, loads it to a DataFrame, drops 
    unnecessary columns and standardizes 'Age' column.
    Finally, it returns the processed DataFrame."""

    df = pd.read_csv(filename)
    df.drop(["Country", "User_ID"], axis=1, inplace=True)
    scaled_X = StandardScaler().fit_transform(df[['Age']])
    df[['Age']] = scaled_X
    return df
    
def plot_elbow_curve(start, end) -> None:
    """Plots elbow curve. Used for optimizing K."""

    data = getUsersDF()
    categorical_index = [1]
    no_of_clusters = list(range(start, end+1))
    cost_values = []
    cores = min(PC, cpu_count())
    
    for k in no_of_clusters:
        print(f"Testing with {k} clusters...")
        test_model = KPrototypes(n_clusters=k, init='Huang', n_init=cores, n_jobs=cores)
        test_model.fit_predict(data, categorical=categorical_index)
        cost_values.append(test_model.cost_)
        
    seaborn.set_theme(style="whitegrid", palette="bright", font_scale=1.1)
    
    plt.figure(figsize=(15, 7))
    ax = seaborn.lineplot(x=no_of_clusters, y=cost_values, marker="o", dashes=False)
    ax.set_title('Elbow curve', fontsize=18)
    ax.set_xlabel('No of clusters', fontsize=14)
    ax.set_ylabel('Cost', fontsize=14)
    ax.set(xlim=(start-0.1, end+0.1))
    plt.plot()
    plt.show()

def kPrototypes() -> pd.DataFrame:
    """Runs k-Prototypes algorithm for Users, placing them
    in one of K clusters. In our case, we used the elbow method
    and found that using K=3 is optimal for this algorithm."""

    dataframe = getUsersDF()
    categorical_features_idx = [1]
    mark_array=dataframe.values
    cores = min(PC, cpu_count())
    test_model = KPrototypes(n_clusters=K, verbose=2, init='Huang', n_init=cores, n_jobs=cores)
    clusters = test_model.fit_predict(mark_array, categorical=categorical_features_idx)

    # Cluster Centroids
    #print(test_model.cluster_centroids_)
    dataframe['Cluster'] = list(clusters)
    dataframe.to_csv("Files/Clustered-Data.csv")

    return dataframe