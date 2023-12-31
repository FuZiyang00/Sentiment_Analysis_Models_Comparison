from sklearn.metrics import classification_report
from tqdm import tqdm 
import numpy as np

class classifiers: 
    def __init__(self, *args):
        self.models = list(args)

    def models_training(self, x_train, y_train, batch_size):
        for model in self.models:
            # training
            with tqdm(total=x_train.shape[0], desc=f"Training {str(model)}") as pbar:
                for i in range(0, x_train.shape[0], batch_size):
                    end_idx = min(i + batch_size, x_train.shape[0])
                    model.fit(x_train[i:end_idx], y_train[i:end_idx])
                    pbar.update(end_idx - i)
    
    def models_evaluation(self, X_test, y_test, output_file):
        with open(output_file, "w") as file:
            for model in self.models:
                # Model evaluation
                test_acc = model.score(X_test, y_test)
                report_str = 'Accuracy score on testing data of {}: {}\n'.format(model, test_acc)
                file.write(report_str)

                y_pred = model.predict(X_test)
                report_str = classification_report(y_test, y_pred, zero_division=1)
                file.write(report_str + '\n')

            print(f"Classification reports saved to {output_file}")
    
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



        
        