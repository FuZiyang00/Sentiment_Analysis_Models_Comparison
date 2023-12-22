from models.lstm import LSTM
from models.rnn import RNN
from data_process.data_cleaner import Data_processor
import pandas as pd
import warnings

class Neural_Networks: 
    @staticmethod
    def Neural_Networks(train_df, val_df, test_df, words_embeddings, outputfile):
        # processsing the training data
        training_data_processor = Data_processor(words_embeddings, train_df)
        X_train, y_train = training_data_processor.columns_processor()
        # determing the maximum lenght of a review 
        sequence_lengths = []
        for i in range(len(X_train)):
            sequence_lengths.append(len(X_train[i]))
        describe_output = pd.Series(sequence_lengths).describe()
        print(describe_output)
        max_length = describe_output['max']
        X_train = training_data_processor.vectors_padding(X_train, max_length)
        print(X_train.shape)

        #processing and padding the validation data 
        validation_data_processor = Data_processor(words_embeddings, val_df)
        X_val, y_val = validation_data_processor.columns_processor()
        X_val = validation_data_processor.vectors_padding(X_val, max_length)
        print(X_val.shape)

        #processing and padding the test data 
        test_data_processor = Data_processor(words_embeddings, test_df)
        X_test, y_test = test_data_processor.columns_processor()
        X_test = test_data_processor.vectors_padding(X_test, max_length)
        print(X_test.shape)

        # one hot encoding the class labels: 
        y_train, y_val, y_test = test_data_processor.label_encoder(y_train, y_val, y_test)
        warnings.simplefilter(action='ignore', category=FutureWarning)
        frequencies = pd.value_counts(train_df['label']) 
        total_samples = frequencies.sum()
        weights = {0: total_samples / frequencies[0], 
                1: total_samples / frequencies[1], 
                2: total_samples / frequencies[2]}
        
        # training the LSTM model 
        LSTM_model = LSTM(X_train, y_train, X_val, y_val, X_test, y_test)
        LSTM_model.model_training(weights)
        LSTM_model.model_evaluation(outputfile)

        # training the RNN model 
        RNN_model = RNN(X_train, y_train, X_val, y_val, X_test, y_test)
        RNN_model.model_training(weights)
        RNN_model.model_evaluation(outputfile)
        