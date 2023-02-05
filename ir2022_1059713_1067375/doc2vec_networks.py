import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from pprint import pprint

#INPUT_FILES = "C:/Users/steli/Documents/Github/Data-Retrieval-Project-2022/Files/Input/"
INPUT_FILES = "Files/Input/"

USERS = INPUT_FILES + "BX-Users.csv"
RATINGS = INPUT_FILES + "BX-Book-Ratings.csv"
BOOKS = INPUT_FILES + "BX-Books.csv"

#PROCESSED_FILES = "C:/Users/steli/Documents/Github/Data-Retrieval-Project-2022/Files/Processed/"
PROCESSED_FILES = "Files/Processed/"
BOOKS_VECTORIZED_SUMMARIES_PKL = PROCESSED_FILES + "Books-Vectorized-Summaries.pkl"
AVG_CLUST_RATING = PROCESSED_FILES + "Average-Cluster-Rating.csv"

def trainNetwork(vect_sums_list, ratings):
    """Trains a regression neural network model to predict average
    cluster ratings of books according to their summary."""
    
    # Prepare input for network
    X = pd.DataFrame(vect_sums_list, columns=["vect"+str(i) for i in range(len(vect_sums_list[0]))])
    Y = np.array(ratings)

    # Check input and label shapes
    print("Input and label shapes:")
    print(X.shape)
    print(Y.shape)

    # Choose model to use
    model = base_model_1()
    model.summary()
    history = model.fit(X, Y, validation_split=0.3, epochs=20, batch_size=10, verbose=1)
    plotHistory(history)

    return model

def custom_loss_function(y_true, y_pred):
   squared_difference = tf.square(10*(y_true - y_pred))
   return tf.reduce_mean(squared_difference, axis=-1)

def base_model_1():
    # Create model
    model = Sequential()
    model.add(Dense(256, input_dim=100, activation='relu'))
    #model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='relu'))
    model.compile(loss=custom_loss_function, optimizer='adam', metrics=["mae"])
    return model

def base_model_2():
    # Insert the dropout layer
    model = Sequential()
    model.add(Dense(1000, input_dim=100, activation='relu')) # (features,)
    model.add(Dropout(0.5)) # specify a percentage between 0 and 0.5, or larger
    model.add(Dense(500, activation='relu'))
    model.add(Dropout(0.5)) # specify a percentage between 0 and 0.5, or larger
    model.add(Dense(250, activation='relu'))
    model.add(Dropout(0.5)) # specify a percentage between 0 and 0.5, or larger
    model.add(Dense(1, activation='linear')) # output node
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=["mae"])
    return model

def trainSingleNetwork(book_ratings: pd.DataFrame, books: pd.DataFrame):
    # Get all vectorized summaries and calculate every book's average ratings
    books_w_ratings = pd.merge(right=book_ratings, left=books, how="left", on="isbn", validate="one_to_many").groupby(by="isbn")
    vect_sums = books_w_ratings.first()["Vectorized_Summary"].to_list()
    ratings = books_w_ratings.mean(numeric_only=True)["rating"].to_list()
    # Train network
    model, history = trainNetwork(vect_sums, ratings)
    plotHistory(history)
    return model
    
def trainClusterNetworks(k: int, avg_clust_ratings: pd.DataFrame, books: pd.DataFrame) -> list[Sequential]:
    models = []
    for i in range(k):
        print(f"Preparing network input for cluster {i}...")
        # Get cluster's vectorized summaries and average rating for all their rated books
        print(books.head())
        cluster_books = pd.merge(right=avg_clust_ratings.loc[avg_clust_ratings["Cluster"]==i], left=books, on="isbn", validate="one_to_one")
        print(cluster_books.head())
        input()
        vect_sums = cluster_books["Vectorized_Summary"].to_list()
        ratings = cluster_books["rating"].to_list()
        print(f"Training network for cluster {i}...")
        models.append(trainNetwork(vect_sums, ratings))

    return models

def trainClusterNetwork(users_cluster: int, vect_books: pd.DataFrame, avg_clust_ratings: pd.DataFrame) -> Sequential:
    print(f"Preparing network input for cluster {users_cluster}...")
    # Get cluster's vectorized summaries and average rating for all their rated books
    
    cluster_books = pd.merge(right=avg_clust_ratings.loc[avg_clust_ratings["Cluster"]==users_cluster], left=vect_books, on="isbn", validate="one_to_one")
    print(cluster_books.head(5))
    vect_sums = cluster_books["Vectorized_Summary"].to_list()
    ratings = cluster_books["rating"].to_list()
    print(f"Training network for cluster {users_cluster}...")
    model = trainNetwork(vect_sums, ratings)

    return model

def plotHistory(history):
    # list all data in history
    #print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['mae'])
    plt.plot(history.history['val_mae'])
    plt.title('model mean absolute error')
    plt.ylabel('mean absolute error')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

def predictClustMissingClusterRatings(cluster: int, avg_clust_ratings: pd.DataFrame, books:pd.DataFrame, model: Sequential):
    predicted_ratings = []
    print(f"Preparing network input for cluster {cluster}...")
    # Get books that have not been rated by cluster
    cluster_missing_ratings = pd.merge(right=avg_clust_ratings.loc[avg_clust_ratings["Cluster"]==cluster], left=books, on="isbn", validate="one_to_one", how="left", indicator=True)
    vect_sums_list = cluster_missing_ratings["Vectorized_Summary"].loc[cluster_missing_ratings["_merge"]=="left_only"].to_list()
    #nn_inputs = [np.array(summ).reshape(1, -1) for summ in vect_sums_list]
    #print(nn_inputs.shape)
    #predicted_ratings.append([models[i].predict(X) for X in nn_inputs])
    print(f"Predicting missing ratings for cluster {cluster}...")
    #for summ in vect_sums_list[:5]:
    #    summ = summ.reshape(1, -1)
    #    print(model.predict(summ))

    predicted_ratings = [model.predict(s.reshape(1, -1))[0][0] for s in vect_sums_list[:5]]

    return predicted_ratings

def runNetworks():
    books_vect_sums = pd.read_pickle(BOOKS_VECTORIZED_SUMMARIES_PKL)
    book_ratings = pd.read_csv(RATINGS)
    avg_clust_ratings = pd.read_csv(AVG_CLUST_RATING)

    #model = trainSingleNetwork(book_ratings, books_vect_sums)
    model = trainClusterNetwork(1, books_vect_sums, avg_clust_ratings)
    # Get book summary (iloc returns list, so summaries
    #clust_miss_ratings = predictClustMissingClusterRatings(1, avg_clust_ratings, books_vect_sums, model)
    #pprint(clust_miss_ratings)
    #books_vect_sums["predicted_rating"] = clust_miss_ratings
    #return clust_miss_ratings

#runNetworks()
#books_vect_sums_pred = runNetworks()
#print(books_vect_sums_pred.head())