from nltk.stem.snowball import SnowballStemmer
import re
from nltk.corpus import stopwords
import string
from emoji import demojize

class Data_Cleaner:
    def __init__(self):
        self.stemmer = SnowballStemmer('english')
        self.regex = re.compile('[%s]' % re.escape(string.punctuation)) # punctuations characters matcher
        self.stopwords = set(stopwords.words('english'))
        self.emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)
    
    def remove_emoji(self, reviews):
        clean_text = self.emoji_pattern.sub(r'', reviews)
        return clean_text

    def special_chars_remover(self, reviews):
        reviews = re.sub(r" +", " ", reviews) # replace one or more consecutive spaces in the text with a single space
        reviews = re.split(r"(\d+|[a-zA-ZğüşıöçĞÜŞİÖÇ]+|\W)", reviews)
        reviews = ' '.join(reviews)
        reviews = self.regex.sub(" ", reviews) # remove punctuations 
        return reviews
    
    def stopwords_remover(self, reviews):
        words = reviews.split()
        clean_text = " ".join([i for i in words if not i in self.stopwords])
        return clean_text
        
    def stemming(self, reviews):
        sentence = [self.stemmer.stem(word) for word in reviews.split()]
        return ' '.join(sentence)
