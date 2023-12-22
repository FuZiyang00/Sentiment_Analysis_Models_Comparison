from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from models.classifiers import classifiers
from data_process.data_cleaner import Data_Cleaner

class Classifiers: 
    @staticmethod
    def classification_methods(X_train, y_train, X_test, y_test, test_sentence):

        tfid = TfidfVectorizer()
        X_train = tfid.fit_transform(X_train)
        X_test = tfid.transform(X_test)
        
        svm_model = SVC(kernel='rbf', probability=True)
        log_model = LogisticRegression()
        knn_model = KNeighborsClassifier()

        classification = classifiers(svm_model, log_model, knn_model)
        classification.models_training(X_train, y_train, 500)
        classification.models_evaluation(X_test, y_test)

        processed_sentence = tfid.transform([Data_Cleaner.text_cleaning(test_sentence)])
        models, predictions, probabilities = classification.review_prediction(processed_sentence)
        for i in range(len(models)):
            print(models[i])
            print("{}: {} with probability {}".format(test_sentence, 
                                                      predictions[i], probabilities[i]))

    