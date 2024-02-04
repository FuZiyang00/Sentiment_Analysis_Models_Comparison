from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm 
import numpy as np
import pandas as pd


class classifiers: 
    def __init__(self, *args):
        self.models = list(args)

    def models_training(self, x_train, y_train):
        for model in self.models:
            model.fit(x_train, y_train)
            print(f"{str(model)} training has been completed")
        
        d = {str(model): model for model in self.models}
        return d
    
    @staticmethod
    def models_evaluation(d, X_test, y_test):
        models_dictionary = {}
        for name, model in d.items(): 
            l = []
            y_pred = model.predict(X_test)
            classification_rep = classification_report(y_test, y_pred)
            conf_matrix = confusion_matrix(y_test, y_pred)
            cm_df = pd.DataFrame(conf_matrix,
                            index = ['Positive','Negative','Neutral'], 
                            columns = ['Positive','Negative','Neutral'])
            l= [classification_rep, cm_df]
            models_dictionary[name]= l
        
        return models_dictionary
    
    # predict the label for a random review (not present in the dataset)
    def review_prediction(self, sentence):
        models = []
        predictions = []
        probabilites = []
        for model in self.models:
            pred_probabilities = model.predict_proba(sentence)
            idx = np.argmax(pred_probabilities)
            pred = model.classes_[idx]
            models.append(model)
            predictions.append(pred)
            probabilites.append(pred_probabilities[0][idx])
        return models, predictions, probabilites 