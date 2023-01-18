import numpy as np
import pandas as pd
from kmodes.kprototypes import KPrototypes
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
import seaborn

# Number of jobs to use for parallelism. 
# Should be maximum the amount of system's CPU Cores
PC = 12

def k_Prototype(dataframe) -> pd.DataFrame:
    categorical_features_idx = [1]
    mark_array=dataframe.values
    test_model = KPrototypes(n_clusters=3, verbose=2, init='Huang', n_init=PC, n_jobs=PC, random_state=42)
    clusters = test_model.fit_predict(mark_array, categorical=categorical_features_idx)

    # Cluster Centroids
    #print(test_model.cluster_centroids_)
    dataframe['Cluster'] = list(clusters)

    return dataframe

# Function for plotting elbow curve
def plot_elbow_curve(start, end, data):
    categorical_index = [0]

    no_of_clusters = list(range(start, end+1))
    cost_values = []

    
    for k in no_of_clusters:
        print(f"Testing with {k} clusters...")
        test_model = KPrototypes(n_clusters=k, init='Huang', n_init=PC, n_jobs=PC, random_state=42)
        test_model.fit_predict(data, categorical=categorical_index)
        cost_values.append(test_model.cost_)
        
    seaborn.set_theme(style="whitegrid", palette="bright", font_scale=1.2)
    
    plt.figure(figsize=(15, 7))
    ax = seaborn.lineplot(x=no_of_clusters, y=cost_values, marker="o", dashes=False)
    ax.set_title('Elbow curve', fontsize=18)
    ax.set_xlabel('No of clusters', fontsize=14)
    ax.set_ylabel('Cost', fontsize=14)
    ax.set(xlim=(start-0.1, end+0.1))
    plt.plot()
    plt.show()
   
def run_k_prototypes() -> pd.DataFrame:
    df = pd.read_csv("Files/BX-Users-BC.csv")
    df.drop(["Country", "User_ID"], axis=1, inplace=True)
    scaled_X = StandardScaler().fit_transform(df[['Age']])
    df[['Age']] = scaled_X

    # Plotting elbow curve for k=1 to k=8
    #plot_elbow_curve(1,8,df.sample(5000))

    clustered_data = k_Prototype(df)
    clustered_data.to_csv("Files/Clustered-Data.csv")
    return clustered_data