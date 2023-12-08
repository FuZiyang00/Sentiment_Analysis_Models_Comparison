from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm

class classifiers: 
    def __init__(self, *args):
        self.models = list(args)

    def models_training(self, x_train, y_train, batch_size):
        for model in self.models:
            with tqdm(total=x_train.shape[0], desc=f"Training {str(model)}") as pbar:
                for i in range(0, x_train.shape[0], batch_size):
                    end_idx = min(i + batch_size, x_train.shape[0])
                    model.fit(x_train[i:end_idx], y_train[i:end_idx])
                    pbar.update(end_idx - i)
    
    def model_evaluation(self, x_test, y_test):
        for model in self.models:
            y_pred = model.predict(x_test)
            
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')

            print(f"{model}:\nAccuracy: {accuracy * 100:.2f}%\nPrecision: {precision:.2f}\nRecall: {recall:.2f}\nF1-score: {f1:.2f}\n")



    

        
        