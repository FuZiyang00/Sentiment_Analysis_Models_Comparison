from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import numpy as np
import string
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelBinarizer
from copy import deepcopy 
from tqdm import tqdm
from multiprocessing import Process, Manager
import pandas as pd

class Text_processor:
    @staticmethod
    def text_cleaning(review):
        clean_text = review.translate(str.maketrans('', '', string.punctuation)).lower()
        clean_text = [word for word in clean_text.split() if word not in stopwords.words('english')]
        sentence = [WordNetLemmatizer().lemmatize(word, 'v') for word in clean_text]
        return ' '.join(sentence)
    
    @staticmethod
    def process_chunks(shared_df, idx):
        chunk = shared_df[idx]
        chunk['Review'] = chunk['Review'].apply(Text_processor.text_cleaning)
        shared_df[idx] = chunk  # Update the shared_df with the processed chunk
        print(f'Chunk {idx} has been processed')

    @staticmethod
    def parallel_text_cleaning(df_chunks): 
        with Manager() as manager:
            shared_df = manager.list(df_chunks) # each process has it's own memory space
            
            processes = []
            for i in range(4):
                process = Process(target=Text_processor.process_chunks, args=(shared_df, i))
                processes.append(process)
                process.start()

            for process in processes:
                process.join()

            cleaned_df = pd.concat(shared_df, ignore_index=True)
        
        return cleaned_df
    
    @staticmethod
    def labelling(x):
        if x == 3:
            return "Neutral"
        elif x<3:
            return "Negative"
        else:
            return "Positive"


class Embeddings_processor:

    def __init__(self, embeddings_dict):
        self.embeddings_dict = embeddings_dict

    def word_vectors(self, review): 
        processed_tokens = review.split()
        vectors = []

        for token in processed_tokens:
            if token not in self.embeddings_dict:
                continue
        
            token_vector = self.embeddings_dict[token]
            vectors.append(token_vector)
        
        return np.array(vectors, dtype=float)
    
    @staticmethod
    def vectors_padding(X_train, sequence_length, batch_size=32):
        num_samples = len(X_train)
        num_batches = (num_samples + batch_size - 1) // batch_size

        X_copy = deepcopy(X_train)

        with tqdm(total=num_samples, desc="Padding Progress") as pbar:
            for batch_num in range(num_batches):
                start_idx = batch_num * batch_size
                end_idx = min((batch_num + 1) * batch_size, num_samples)

                batch_X = X_train[start_idx:end_idx]

                for i, x in enumerate(batch_X):
                    x_seq_len = x.shape[0]

                    if x_seq_len < sequence_length:
                        sequence_length_difference = sequence_length - x_seq_len
                        zero_pad = np.zeros(shape=(sequence_length_difference, 50), dtype=np.float32)
                        X_copy[start_idx + i] = np.concatenate([x, zero_pad])
                    elif x_seq_len > sequence_length:
                        X_copy[start_idx + i] = x[:sequence_length]
                    else:
                        X_copy[start_idx + i] = x

                pbar.update(1)

        return np.array(X_copy).astype(np.float32)
    
    @staticmethod
    def label_encoder(y_train, y_val, y_test):
        label_binarizer = LabelBinarizer()
        encoded_y_train = label_binarizer.fit_transform(y_train) 
        encoded_y_val = label_binarizer.transform(y_val)
        encoded_y_test = label_binarizer.transform(y_test)

        return encoded_y_train, encoded_y_val, encoded_y_test