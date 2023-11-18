import pandas as pd 
from data_process.data_preprocess import EDA
from data_process.data_cleaner import Data_Cleaner
from plotly.offline import plot
from tqdm import tqdm
import re, string

if __name__ == "__main__":
    df = pd.read_csv("tripadvisor_hotel_reviews.csv")
    ratings = "Rating"
    reviews = "Review"
    print(df.head(10), "\n")
    tqdm.pandas()

    print("Eploratory Data Analysis: ")
    eda = EDA(df)
    df["label"] = df[ratings].apply(lambda x: eda.labelling(x))
    shape, duplicates, info = eda.dataset_info()
    statistics, figure = eda.summary_statistics(ratings)
    #line_plot, scatter_plot = eda.variables_relationship(reviews)
    print(df.head(10), "\n")

    print("setence cleaning")
    cleaner = Data_Cleaner()
    def clean_text(x):
        cleaner = Data_Cleaner()
        return cleaner.stemming(cleaner.stopwords_remover(
                                        cleaner.special_chars_remover(x)
        ))
    
    df[reviews] = df[reviews].progress_apply(clean_text)
    print(df.head(10))


    

    
    