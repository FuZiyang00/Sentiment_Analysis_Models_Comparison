import pandas as pd 
from data_process.data_exploration import EDA
from data_process.data_cleaner import Data_Cleaner, RNN_Data_Process, LSTM_data_process
from models.classifiers import classifiers
from models.rnn import RNN_model
from models.lstm import LSTM
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import requests
import zipfile
import os 
import numpy as np
import warnings


if __name__ == "__main__":

    print("Load cleaned df")
    df = pd.read_csv("clean_reviews.csv")
    tqdm.pandas()
    print(df.head(10), "\n")
    eda = EDA(df)
    df["label"] = df["Rating"].apply(lambda x: eda.labelling(x))

    print("Loading the word embeddings")
    words_embeddings = dict()
    # Explicitly specify the encoding as 'utf-8'
    with open('glove.6B.50d.txt', 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = line.split(' ')
            try:
                words_embeddings[line[0]] = np.array(line[1:], dtype=float)
            except Exception as e:
                # Print the exception for debugging purposes
                print(f"Error processing line: {line}. Exception: {e}")
                continue

    print("Splitting the dataset")
    # shuffle the rows of the df and reset the index of the resuting df
    train_df = df.sample(frac=1, random_state=1)
    train_df.reset_index(drop=True, inplace=True)

    split_index_1 = int(len(train_df) * 0.7)
    split_index_2 = int(len(train_df) * 0.85)

    train_df, val_df, test_df = train_df[:split_index_1], train_df[split_index_1:split_index_2], train_df[split_index_2:]

    training_data_processor = LSTM_data_process(words_embeddings, train_df)
    X_train, y_train = training_data_processor.columns_processor()

    # determing the maximum lenght of a review 
    sequence_lengths = []
    for i in range(len(X_train)):
        sequence_lengths.append(len(X_train[i]))

    print(pd.Series(sequence_lengths).describe())

    # padding the training set based on the maximum lenght of a review
    max_length = 1604
    X_train = training_data_processor.vectors_padding(X_train, max_length)
    print(X_train.shape)

    print("validation set")
    validation_data_processor = LSTM_data_process(words_embeddings, val_df)
    X_val, y_val = validation_data_processor.columns_processor()
    X_val = validation_data_processor.vectors_padding(X_val, max_length)
    print(X_val.shape)

    print("test set")
    test_data_processor = LSTM_data_process(words_embeddings, test_df)
    X_test, y_test = test_data_processor.columns_processor()
    X_test = test_data_processor.vectors_padding(X_test, max_length)
    print(X_test.shape)

    # Before the code that produces warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)
    frequencies = pd.value_counts(train_df['label']) 
    total_samples = frequencies.sum()
    weights = {0: total_samples / frequencies[0], 
               1: total_samples / frequencies[1], 
               2: total_samples / frequencies[2]}
    
    LSTM_model = LSTM(X_train, y_train, X_val, y_val, X_test, y_test)
    LSTM_model.model_training(weights)
    LSTM_model.model_evaluation()
    warnings.simplefilter(action='default', category=FutureWarning)
    
    



    
    
    
    

    """
    df = pd.read_csv("tripadvisor_hotel_reviews.csv")
    print("Text cleaning")
    df[reviews] = df[reviews].progress_apply(Data_Cleaner.text_cleaning)
    df.to_csv('clean_reviews.csv', index='False')
    """
    
    """
    print("Training and testing Classifiers")
    tfid = TfidfVectorizer()
    # Fit and transform on the training set
    with tqdm(total=len(x_train), desc="Fitting and transforming train set") as pbar:
        train_tfid_matrix = tfid.fit_transform(x_train)
        pbar.update(len(x_train))

    # Transform on the test set
    with tqdm(total=len(x_test), desc="Transforming test set") as pbar:
        test_tfid_matrix = tfid.transform(x_test)
        pbar.update(len(x_test))
    
    print("training and testing the classifiers")
    svm_model = SVC(kernel='rbf', probability=True)
    log_model = LogisticRegression()
    knn_model = KNeighborsClassifier()

    # Training and evaluating the performances of the selected model(s)
    classification = classifiers(svm_model, log_model, knn_model)
    classification.models_training_evaluation(train_tfid_matrix, y_train, test_tfid_matrix, y_test, 500)

    # testing the models on a random review
    sentence = "This hotel is worth every penny!!"
    processed_sentence = tfid.transform([Data_Cleaner.text_cleaning(sentence)])
    models, predictions, probabilities = classification.review_prediction(processed_sentence)
    for i in range(len(models)):
        print(models[i])
        print("{}: {} with probability {}".format(sentence, predictions[i], probabilities[i]))

    # Training and evaluating the performances of the Recurrent Neural Network 
    print("Training and testing RNN")
    data_processor = RNN_Data_Process()

    total_word, train_padded = data_processor.training_tokenizer(x_train)
    test_padded = data_processor.testing_tokenizer(x_test)
    train_labels, test_labels = data_processor.label_encoder(y_train, y_test)

    model = RNN_model.create_model(total_word)
    model.fit(train_padded, train_labels, 
              epochs=1, validation_data=(test_padded, test_labels))
    RNN_model.model_evaluation(test_padded, test_labels, model)

    # testing the model on a random review
    sentence = "I don't even want a refund from this place..."
    processsed_sentence = data_processor.testing_tokenizer([Data_Cleaner.text_cleaning(sentence)])
    prediction = model.predict(processsed_sentence)
    result = data_processor.label_binarizer.inverse_transform(prediction)[0]
    print("{}: {}".format(sentence, result))
    """

    """
    url = "http://downloads.cs.stanford.edu/nlp/data/glove.6B.zip" 
    response = requests.get(url)

    with open("glove.6B.zip", "wb") as f:
        f.write(response.content)

    # Specify the path to the downloaded zip file
    zip_file_path = "glove.6B.zip"

    # Specify the directory where you want to extract the contents (current working directory)
    extracted_dir = "."

    # Open the zip file
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        # Extract all the contents into the target directory
        zip_ref.extractall(extracted_dir)

    print(f"Files extracted to {extracted_dir}")
    """
    





    



    

    
    