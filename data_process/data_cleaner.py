from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
import string
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelBinarizer
from keras.preprocessing.text import Tokenizer

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
    
class RNN_Data_Process:
    def __init__(self):
        # Initialize Tokenizer and LabelBinarizer
        self.tokenizer = Tokenizer(num_words=50000, oov_token='<OOV>')
        self.label_binarizer = LabelBinarizer()

    def training_tokenizer(self, x_train):
        # Fit the tokenizer on training data
        self.tokenizer.fit_on_texts(x_train)
        
        # Get the total number of unique words in the training data
        total_words = len(self.tokenizer.word_index)
        
        # Convert texts to sequences and pad them
        train_seq = self.tokenizer.texts_to_sequences(x_train)
        train_padded = pad_sequences(train_seq)

        return total_words, train_padded

    def testing_tokenizer(self, x_test):
        # Convert test texts to sequences and pad them
        test_seq = self.tokenizer.texts_to_sequences(x_test)
        test_padded = pad_sequences(test_seq)

        return test_padded

    def label_encoder(self, y_train, y_test):
        # Fit and transform labels for training data
        train_labels = self.label_binarizer.fit_transform(y_train)
        
        # Transform labels for test data
        test_labels = self.label_binarizer.transform(y_test)

        return train_labels, test_labels


                