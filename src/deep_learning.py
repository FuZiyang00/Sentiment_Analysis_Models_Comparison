from sklearn.metrics import classification_report, confusion_matrix
from keras import layers
from keras.models import Sequential 
from keras.optimizers import Adam
from keras.metrics import AUC
import numpy as np
import pandas as pd

class BaseModel:

    def __init__(self, X_train, y_train, X_val, y_val, X_test, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.X_val = X_val
        self.y_train = y_train
        self.y_test = y_test
        self.y_val = y_val
        self.model = self.model_creation()

    def model_creation(self):
        raise NotImplementedError("Subclasses must implement this method.")

    def model_training(self, epochs):
        self.model.fit(self.X_train, self.y_train, 
                  validation_data=(self.X_val, self.y_val), epochs=epochs)
    
    def model_evaluation(self, y_test):
        test_predictions = self.model.predict(self.X_test)
        decoded_predictions = np.argmax(test_predictions, axis=1)
        report_str = classification_report(np.argmax(self.y_test, axis=1), decoded_predictions, zero_division=1)
        conf_matrix = confusion_matrix(y_test, decoded_predictions)
        cm_df = pd.DataFrame(conf_matrix,
                            index = ['Positive','Neutral','Negative'], 
                            columns = ['Positive','Neutral','Negative'])
        
        
        return report_str, cm_df

class LSTM(BaseModel): 

    def model_creation(self):
        model = Sequential([])
        model.add(layers.Input(shape=(1807, 50)))
        model.add(layers.LSTM(64, return_sequences=True))
        model.add(layers.Dropout(0.2))
        model.add(layers.Flatten())
        model.add(layers.Dense(3, activation='softmax'))

        model.compile(optimizer=Adam(learning_rate=0.0001), 
              loss='categorical_crossentropy', 
              metrics=['accuracy', AUC(name='auc')])
        
        return model 

class RNN(BaseModel):

    def model_creation(self):
        model = Sequential([])
        model.add(layers.Input(shape=(1807, 50)))
        model.add(layers.SimpleRNN(64, return_sequences=False))
        model.add(layers.Dense(3, activation='softmax'))
        
        model.compile(optimizer=Adam(learning_rate=0.0001), 
              loss='categorical_crossentropy', 
              metrics=['accuracy', AUC(name='auc')])
        
        return model 