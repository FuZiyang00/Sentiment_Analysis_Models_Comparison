from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
import string

from sklearn.preprocessing import LabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, classification_report
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences



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
    
class RNN_Data_process:
    def __init__(self, Tokenizer, LabelBinarizer):
        self.tokenizer = Tokenizer(num_words=50000, oov_token='<OOV>')
        self.labels = LabelBinarizer()
    
    def training_tokenizer(self, x_train):
        self.tokenizer(x_train)
        total_word = len(self.tokenizer.word_index)
        train_seq = self.tokenizer.texts_to_sequences(x_train)
        train_padded = pad_sequences(train_seq) 

        return total_word, train_padded
    
    def testing_tokenizer(self, x_test):
        test_seq = self.tokenizer.texts_to_sequences(x_test)
        test_padded = pad_sequences(test_seq)

        return test_padded
    
    def label_encoder(self, y_train, y_test):
        train_labels = self.labels.fit_transform(y_train)
        test_labels = self.labels.transform(y_test)

        return train_labels, test_labels


                