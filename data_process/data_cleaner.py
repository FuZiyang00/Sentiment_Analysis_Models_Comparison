import nltk
nltk.download('wordnet')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import numpy as np
import string
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelBinarizer
from keras.preprocessing.text import Tokenizer
from copy import deepcopy 
from tqdm import tqdm

class Data_Cleaner:
    @staticmethod
    def text_cleaning(review):
        #remove punctuations and uppercase
        clean_text = review.translate(str.maketrans('','',string.punctuation)).lower()
        
        #remove stopwords
        clean_text = [word for word in clean_text.split() if word not in stopwords.words('english')]
        
        #lemmatize the word
        sentence = []
        for word in clean_text:
            lemmatizer = WordNetLemmatizer()
            sentence.append(lemmatizer.lemmatize(word, 'v'))

        return ' '.join(sentence)
    
    # Split the DataFrame into chunks for parallel processing
    @staticmethod
    def split_dataframe(df, num_chunks):
        chunk_size = len(df) // num_chunks
        return [df.iloc[i:i + chunk_size] for i in range(0, len(df), chunk_size)]

    # Function to apply text cleaning to a DataFrame chunk
    def process_chunk(chunk):
        chunk['Review'] = chunk['Review'].progress_apply(Data_Cleaner.text_cleaning)
        return chunk

class Data_processor:

    def __init__(self, embeddings_dict, df):
        self.embeddings_dict = embeddings_dict
        self.df = df

    def word_vectors(self, review): 
        processed_tokens = review.split()
        vectors = []

        for token in processed_tokens:
            if token not in self.embeddings_dict:
                continue
        
            token_vector = self.embeddings_dict[token]
            vectors.append(token_vector)
        
        return np.array(vectors, dtype=float)
    
    def columns_processor(self):
        label_mapping = {'Positive': 2, 'Negative': 0, "Neutral":1}
        self.df['label'] = self.df['label'].map(label_mapping)
        y = self.df['label'].to_numpy().astype(int)
        all_word_vector_sequences = []

        for message in self.df['Review']:
            message_as_vector_seq = self.word_vectors(message)
            
            if message_as_vector_seq.shape[0] == 0:
                message_as_vector_seq = np.zeros(shape=(1, 50))

            all_word_vector_sequences.append(message_as_vector_seq)
        
        return all_word_vector_sequences, y
    
    def vectors_padding(self, X_train, sequence_length, batch_size=32):
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
         




                