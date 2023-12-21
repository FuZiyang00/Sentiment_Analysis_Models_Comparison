from keras import layers
from keras.models import Sequential 
from keras.optimizers import Adam
from keras.metrics import AUC
from sklearn.metrics import classification_report

class LSTM: 

    def __init__(self, X_train, y_train, X_val, y_val, X_test, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.X_val = X_val
        self.y_train = y_train
        self.y_test = y_test
        self.y_val = y_val
        self.model = self.model_creation()

    def model_creation(self):
        model = Sequential([])
        model.add(layers.Input(shape=(1604, 50)))
        model.add(layers.LSTM(64, return_sequences=True))
        model.add(layers.Dropout(0.2))
        model.add(layers.LSTM(64, return_sequences=True))
        model.add(layers.Dropout(0.2))
        model.add(layers.Flatten())
        model.add(layers.Dense(1, activation='sigmoid'))

        model.compile(optimizer=Adam(learning_rate=0.0001), 
              loss='categorical_crossentropy', 
              metrics=['accuracy', AUC(name='auc')])
        
        return model 
    
    def model_training(self, class_weights):
        self.model.fit(self.X_train, self.y_train, 
                  validation_data=(self.X_val, self.y_val), 
                  epochs=5, class_weight=class_weights)
    
    def model_evaluation(self):
        test_predictions = self.model.predict(self.X_test)
        print(classification_report(self.y_test, test_predictions))
