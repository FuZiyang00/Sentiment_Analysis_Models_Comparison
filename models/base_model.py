from sklearn.metrics import classification_report

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

    def model_training(self, class_weights):
        self.model.fit(self.X_train, self.y_train, 
                  validation_data=(self.X_val, self.y_val), 
                  epochs=5, class_weight=class_weights)
    
    def model_evaluation(self, output_file):
        with open(output_file, "w") as file:
            test_predictions = self.model.predict(self.X_test)
            report_str = classification_report(self.y_test, test_predictions)
            file.write(report_str + '\n')