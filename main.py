import pandas as pd 
from data_process.data_preprocess import EDA

if __name__ == "__main__":
    reviews = pd.read_csv("tripadvisor_hotel_reviews.csv")
    print(reviews.head(10))

    print("Eploratory Data Analysis")
    eda = EDA(reviews)
    shape, duplicates, info = eda.dataset_info()
    print(shape)    
    print(duplicates)
    print(info)
    column_name1 = "Rating"
    statistics, figure = eda.summary_statistics(column_name1)
    print(statistics)