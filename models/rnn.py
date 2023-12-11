import tensorflow as tf 
from keras.models import Sequential
from keras.layers import SimpleRNN, Dense, Embedding
import numpy as np
from sklearn.metrics import classification_report
from data_process.data_cleaner import RNN_Data_Process, Data_Cleaner
from sklearn.preprocessing import LabelBinarizer


class RNN_model:
    @staticmethod
    def create_model(total_words):
        model = Sequential([Embedding(total_words, 8),
                                    SimpleRNN(16),
                                    Dense(3, activation='softmax')])
        
        model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.0001), 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
        
        return model 
    
    @staticmethod
    def model_evaluation(test_padded, test_labels, model):
        predictions = model.predict(test_padded)
        true_labels = np.argmax(test_labels, axis=-1)
        pred_labels = np.argmax(predictions, axis=-1)
        print(classification_report(true_labels, pred_labels))