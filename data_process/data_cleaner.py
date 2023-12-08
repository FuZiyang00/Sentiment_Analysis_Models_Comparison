from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
import string

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