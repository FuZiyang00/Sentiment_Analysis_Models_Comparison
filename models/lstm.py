from keras import layers
from keras.models import Sequential 
from keras.optimizers import Adam
from keras.metrics import AUC
from .base_model import BaseModel

class LSTM(BaseModel): 

    def model_creation(self):
        model = Sequential([])
        model.add(layers.Input(shape=(1604, 50)))
        model.add(layers.LSTM(64, return_sequences=True))
        model.add(layers.Dropout(0.2))
        model.add(layers.Flatten())
        model.add(layers.Dense(3, activation='softmax'))

        model.compile(optimizer=Adam(learning_rate=0.0001), 
              loss='categorical_crossentropy', 
              metrics=['accuracy', AUC(name='auc')])
        
        return model 
    
    

