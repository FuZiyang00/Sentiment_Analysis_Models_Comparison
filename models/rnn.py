from keras import layers
from keras.optimizers import Adam
from keras.metrics import AUC
from keras.models import Sequential
from sklearn.metrics import classification_report
from .base_model import BaseModel

class RNN(BaseModel):

    def model_creation(self):
        model = Sequential([])
        model.add(layers.Input(shape=(1604, 50)))
        model.add(layers.SimpleRNN(64, return_sequences=False))
        model.add(layers.Dense(3, activation='softmax'))
        
        model.compile(optimizer=Adam(learning_rate=0.0001), 
              loss='categorical_crossentropy', 
              metrics=['accuracy', AUC(name='auc')])
        
        return model 