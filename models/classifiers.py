from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from tqdm import tqdm

class classifiers: 
    def __init__(self, *args):
        self.models = list(args)

    def models_training(self, x_train, y_train, batch_size=100):
        for model in self.models:
            # Create a tqdm progress bar for the current model
            with tqdm(total=x_train.shape[0], desc=f"Training {str(model)}") as pbar:
                for i in range(0, x_train.shape[0], batch_size):
                    end_idx = min(i + batch_size, x_train.shape[0])
                    model.fit(x_train[i:end_idx], y_train[i:end_idx])
                    pbar.update(end_idx - i)
    
    def model_evaluation(self, x_test, y_test):
        for model in self.models:
            y_pred = model.predict(x_test)
            print(model,":",accuracy_score(y_test,y_pred)*100)



    

        
        