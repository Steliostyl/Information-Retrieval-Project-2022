import numpy as np
import pandas as pd
from kmodes.kprototypes import KPrototypes

def k_Prototype(filename: str = "Files/BX-Users-BC.csv"):
    df = pd.read_csv(filename)
    print(df.shape)
    df.head()
    categorical_features_idx = [0]
    mark_array=df.values
    kproto = KPrototypes(n_clusters=5, verbose=2, max_iter=20).fit(mark_array, categorical=categorical_features_idx)

    # Cluster Centroids
    print(kproto.cluster_centroids_)
    # Prediction
    clusters = kproto.predict(mark_array, categorical=categorical_features_idx)
    df['cluster'] = list(clusters)

    # Cluster 1 samples
    df[df['cluster']== 0].head(10)
    # Cluster 2 samples
    df[df['cluster']==1].head(10)


k_Prototype()