import pandas as pd 

if __name__ == "__main__":
    reviews = pd.read_csv("tripadvisor_hotel_reviews.csv")
    print(reviews.head(10))

    print("Eploratory Data Analysis")
    