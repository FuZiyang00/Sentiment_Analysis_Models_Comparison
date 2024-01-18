import pandas as pd
from data_process.data_cleaner import Data_Cleaner
from multiprocessing import Pool
import logging
from tqdm import tqdm 
from data_process.data_exploration import EDA
from sklearn.model_selection import train_test_split
from training_evaluation_scripts.classifiers_training import Classifiers
from training_evaluation_scripts.lstm_rnn import Neural_Networks
import numpy as np
import requests
import zipfile
import tensorflow as tf

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# Set TensorFlow logging level to suppress warnings
tf.get_logger().setLevel(logging.ERROR)

if __name__ == "__main__":

    logger.info("Loading data and cleaning the reviews")
    tqdm.pandas()
    df = pd.read_csv("tripadvisor_hotel_reviews.csv")
    df_chunks = Data_Cleaner.split_dataframe(df, 4)

    # Use a Pool to parallelize the processing
    with Pool(4) as pool:
        processed_chunks = list(tqdm(pool.imap(Data_Cleaner.process_chunk, df_chunks), total=4))

    # Concatenate the processed chunks back into a DataFrame
    df = pd.concat(processed_chunks, ignore_index=True)
    
    print(df.head(10), "\n")
    output_file = "report.txt"
    
    eda = EDA(df)
    df["label"] = df["Rating"].apply(lambda x: eda.labelling(x))

    logger.info("Using classification techniques")
    X_train, X_test, y_train, y_test = train_test_split(df['Review'], df['label'], test_size=0.2)
    test_sentence = "I love how this hotel, but the furitures could be better."
    Classifiers.classification_methods(X_train, y_train, X_test, y_test, 
                                       output_file,test_sentence)

    logger.info("Using RNN and LSTM")
    logger.info("Importing pretrained word embeddings")
    
    url = "http://downloads.cs.stanford.edu/nlp/data/glove.6B.zip" 
    response = requests.get(url)

    with open("glove.6B.zip", "wb") as f:
        f.write(response.content)

    zip_file_path = "glove.6B.zip"
    # directory where to extract the contents (current working directory)
    extracted_dir = "."

    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        # Extract all the contents into the target directory
        zip_ref.extractall(extracted_dir)
    logger.info(f"Files extracted to {extracted_dir}")
    
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
    
    df = df.sample(frac=1, random_state=1)
    df.reset_index(drop=True, inplace=True)
    split_index_1 = int(len(df) * 0.7)
    split_index_2 = int(len(df) * 0.85)
    train_df, val_df, test_df = df[:split_index_1], df[split_index_1:split_index_2], df[split_index_2:]
    Neural_Networks.Neural_Networks(train_df, val_df, test_df, words_embeddings, output_file)

    #TODO: 3. !important class weighting methods (check statistics class slides) 

