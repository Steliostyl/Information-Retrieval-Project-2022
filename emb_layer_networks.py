from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Flatten,Embedding,Dense, Dropout
import numpy as np
from gensim.utils import tokenize
import html
import pandas as pd
import re
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.optimizers import adam_v2

INPUT_FILES = "C:/Users/steli/Documents/Github/Data-Retrieval-Project-2022/Files/Input/"

USERS = INPUT_FILES + "BX-Users.csv"
RATINGS = INPUT_FILES + "BX-Book-Ratings.csv"
BOOKS = INPUT_FILES + "BX-Books.csv"

PROCESSED_FILES = "C:/Users/steli/Documents/Github/Data-Retrieval-Project-2022/Files/Processed/"
BOOKS_VECTORIZED_SUMMARIES_PKL = PROCESSED_FILES + "Books-Vectorized-Summaries.pkl"
AVG_CLUST_RATING = PROCESSED_FILES + "Average-Cluster-Rating.csv"

VOCAB_SIZE = 86_982
MAX_LENGTH = 52

def getAllSummariesAndRatings(books: pd.DataFrame, book_ratings: pd.DataFrame) -> tuple:
    books_w_ratings = pd.merge(right=book_ratings, left=books, how="inner", on="isbn", validate="one_to_many").dropna().groupby(by="isbn")
    summaries = books_w_ratings.first()["summary"].to_list()
    ratings = books_w_ratings.mean(numeric_only=False)["rating"].to_list()
    return (summaries, ratings)

def getClusterSummariesAndRatings(books:pd.DataFrame, avg_clust_ratings: pd.DataFrame, i):
    cluster_ratings = pd.merge(right=avg_clust_ratings.loc[avg_clust_ratings["Cluster"]==i], left=books, on="isbn", validate="one_to_one")
    summaries = cluster_ratings["summary"].to_list()
    ratings = cluster_ratings["rating"].to_list()
    return (summaries, ratings)

def getNetworkInput(summaries: list, ratings: list, vocab_size, max_length, classifier = False):
    # Preprocess summaries
    preproc_summaries = [preProcessSummaryv3(summary) for summary in summaries]
    # One hot encode words of documents
    encoded_sums = [one_hot(d,vocab_size) for d in preproc_summaries]
    # Add padding
    X = pad_sequences(encoded_sums,maxlen=max_length,padding='post')
    # Normalize ratings
    if classifier:
        Y = np.array([customOHE2(r) for r in ratings])
        return (X, Y, classifier_model_1(vocab_size, max_length))
    # Define the model
    model = base_model_1(vocab_size, max_length)
    Y = np.array(ratings)
    return (X, Y, model)

def preProcessSummaryv2(summary) -> list:
    return list(tokenize(html.unescape(summary), lowercase=True, deacc=True))

def preProcessSummaryv3(summary) -> str:
    return re.sub('[^A-Za-z0-9]+', ' ', html.unescape(summary))

#def trainNetwork(X: np.array, Y: np.array, model: Sequential):
#    print(X.shape)
#    print(Y.shape)
#
#    #model = base_model_2(X.shape[1])
#    model.summary()
#    history = model.fit(X, Y, validation_split=0.3, epochs=10, batch_size=10, verbose=1)
#    return (model, history)

def base_model_1(vocab_size, max_length):
    max_length = max_length
    # Create model
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=128,
                        input_length=max_length))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='relu'))
    model.compile(optimizer='adam',loss="mse",metrics=['mae'])
    return model

def base_model_2(vocab_size, max_length):
    # Insert the dropout layer
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=512,input_length=max_length))
    model.add(Flatten())
    model.add(Dropout(0.5)) # specify a percentage between 0 and 0.5, or larger
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5)) # specify a percentage between 0 and 0.5, or larger
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5)) # specify a percentage between 0 and 0.5, or larger
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5)) # specify a percentage between 0 and 0.5, or larger
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5)) # specify a percentage between 0 and 0.5, or larger
    model.add(Dense(1, activation='linear')) # output node
    model.compile(optimizer='adam',loss=custom_loss_function,metrics=['mean_absolute_error'])
    return model

def custom_loss_function(y_true, y_pred):
   squared_difference = tf.square(10*(y_true - y_pred))
   return tf.reduce_mean(squared_difference, axis=-1)

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

def trainClusterNetwork(cluster: int, avg_clust_ratings: pd.DataFrame,
                        books: pd.DataFrame, vocab_size: int, max_length: int):
    """Train regression NN on user’s cluster’s average ratings"""

    print(f"Preparing network input for cluster {cluster}...")
    # Add book summaries to the avg_clust_ratings df
    print(avg_clust_ratings.loc[avg_clust_ratings["Cluster"]==cluster].head())
    cluster_ratings = pd.merge(
        right=avg_clust_ratings.loc[avg_clust_ratings["Cluster"]==cluster],
        left=books, on="isbn", validate="one_to_one")
    summaries = cluster_ratings["summary"].to_list()
    ratings = cluster_ratings["rating"].to_list()

    X, Y, model = getNetworkInput(summaries, ratings, vocab_size, max_length)
    print(f"Training network for cluster {cluster}...")

    # Print input and label shapes
    print(X.shape)
    print(Y.shape)
    # Print model summary
    model.summary()
    # Start training
    history = model.fit(X, Y, validation_split=0.3, epochs=10,
                        batch_size=10, verbose=1)
    plotHistory(history)

    return model

def trainSingleNetwork(books, book_ratings, vocab_size: int, max_length: int, classifier = False):
    summaries, ratings = getAllSummariesAndRatings(books, book_ratings)
    X, Y, model = getNetworkInput(summaries, ratings,  vocab_size=vocab_size, max_length=max_length, classifier=classifier)
    # Print input and label shapes
    print(X.shape)
    print(Y.shape)
    # Print model summary
    model.summary()
    # Start training
    history = model.fit(X, Y, validation_split=0.3, epochs=10, batch_size=10, verbose=1)
    plotHistory(history)
    
    plotHistory(history)
    return model

def calculateVocab(books: pd.DataFrame):
    """Calculates the vocabulary size and the
    max length of the rated books of a cluster"""
    summaries = [preProcessSummaryv3(book[1]["summary"])
                 for book in books.iterrows()]
    vocab_size = 0
    max_length = 0
    vocab = {}
    for summary in summaries:
        words = summary.split()
        max_length = max(max_length, len(words))
        for word in words:
            if word not in vocab:
                vocab[word] = 0
    vocab_size = len(vocab)
    return vocab_size, max_length

def customOHE2(rating):
    if rating > 8:
        return [0, 0, 0, 0, 1]
    elif rating > 6:
        return [0, 0, 0, 1, 0]
    elif rating > 4:
        return [0, 0, 1, 0, 0]
    elif rating > 2:
        return [0, 1, 0, 0, 0]
    else:
        return [1, 0, 0, 0, 0]
    
def classifier_model_1(vocab_size, max_length):
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=256,input_length=max_length))
    model.add(Flatten())
    model.add(Dense(256, activation="relu"))
    model.add(Dense(128, activation="relu"))
    model.add(Dense(5, activation='softmax'))
    model.compile(optimizer='adam',loss="binary_crossentropy",metrics=['acc'])
    return model

def predictClustMissingRatings(k: int, avg_clust_ratings: pd.DataFrame, books:pd.DataFrame, models: list[Sequential], vocab_size, max_length):
    predicted_ratings = []
    for i in range(k):
        print(f"Preparing network input for cluster {i}...")
        # Get books that have not been rated by cluster
        cluster_ratings = pd.merge(right=avg_clust_ratings.loc[avg_clust_ratings["Cluster"]==i], left=books, on="isbn", validate="one_to_one", how="left", indicator=True)
        summaries = cluster_ratings["summary"].loc[cluster_ratings["_merge"]=="left_only"].to_list()
        ratings = [0 for i in range(len(summaries))]

        X, _, _ = getNetworkInput(summaries, ratings, vocab_size=vocab_size, max_length=max_length)
        print(X.shape)
        print(X)
        predicted_ratings.append(models[i].predict(X))

    return predicted_ratings

def runNetworks():
    books = pd.read_csv(BOOKS)
    book_ratings = pd.read_csv(RATINGS)
    avg_clust_ratings = pd.read_csv(AVG_CLUST_RATING)

    print(f"Calculating vocab size...")
    vs, ml = calculateVocab(books)
    print(f"Vocab size: {vs}, Max length: {ml}")
    #model = trainSingleNetwork(books, book_ratings, vocab_size=vs, max_length=ml, classifier=False)
    model = trainClusterNetwork(1, avg_clust_ratings, books, vocab_size=vs, max_length=ml)
    ## Get book summary (iloc returns list, so summaries
    #clust_miss_ratings = predictClustMissingRatings(1, avg_clust_ratings, books, models)
    #print(clust_miss_ratings[:5])

def checkRatings():
    book_ratings = pd.read_csv(RATINGS)
    for r in book_ratings["rating"]:
        try:
            rat = float(r)
            if rat < 0 or rat >10:
                print(rat)
        except:
            print("Not a number!")