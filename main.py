import pandas as pd 
from data_process.data_exploration import EDA
from data_process.data_cleaner import Data_Cleaner
from models.classifiers import classifiers
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


if __name__ == "__main__":
    df = pd.read_csv("tripadvisor_hotel_reviews.csv")
    ratings = "Rating"
    reviews = "Review"
    # print(df.head(10), "\n")
    tqdm.pandas()

    print("Eploratory Data Analysis: ")
    eda = EDA(df)
    df["label"] = df[ratings].apply(lambda x: eda.labelling(x))
    shape, duplicates, info = eda.dataset_info()
    statistics, figure = eda.summary_statistics(ratings)
    #line_plot, scatter_plot = eda.variables_relationship(reviews)
    print(df.head(10), "\n")
    
    print("Text cleaning")
    df[reviews] = df[reviews].progress_apply(Data_Cleaner.text_cleaning)
    print(df.head(10))
    
    print("Training and testing")
    x_train, x_test, y_train, y_test = train_test_split(df[reviews], df["label"], test_size=0.3, random_state=42)

    tfid = TfidfVectorizer()
    # Fit and transform on the training set
    with tqdm(total=len(x_train), desc="Fitting and transforming train set") as pbar:
        train_tfid_matrix = tfid.fit_transform(x_train)
        pbar.update(len(x_train))

    # Transform on the test set
    with tqdm(total=len(x_test), desc="Transforming test set") as pbar:
        test_tfid_matrix = tfid.transform(x_test)
        pbar.update(len(x_test))
    
    print("training and testing the classifiers")
    svm_model = SVC(kernel='rbf', probability=True)
    log_model = LogisticRegression(probability = True)
    knn_model = KNeighborsClassifier(probability = True)

    classification = classifiers(svm_model, log_model, knn_model)
    classification.models_training_evaluation(train_tfid_matrix, y_train, test_tfid_matrix, y_test, 500)
    sentence = tfid.transform([Data_Cleaner.text_cleaning("This hotel is worth every penny!!")])
    classification.review_prediction(sentence)
    



    

    
    