![Alt text](https://d3caycb064h6u1.cloudfront.net/wp-content/uploads/2021/06/sentimentanalysishotelgeneric-2048x803-1.jpg)

# Sentiment Analysis Models Comparison
The goal of this project is to analyze the performance of different machine learning and deep learning models in predicting sentiment related to TripAdvisor reviews. 

The tested models were the following: SVM, K Nearest neighbours, Logistic Regression, RNN, LSTM.

The employed dataset can be found at the following link: https://www.kaggle.com/datasets/andrewmvd/trip-advisor-hotel-reviews 

## Performances
-------
| MODEL|PRECISION| RECALL| F-SCORE|
| :-------------:|:-------------:| :-------------:|:-------------:|
| **SVM**      | 0.73| 0.72 |0.72|
|**KNEIGHBOURS**|0.56|0.56|0.56|
|**LOGISTIC REGRESSION**|0.71|0.71|0.71|
|**RNN**|0.46|0.33|0.48|
|**LSTM**|0.67|0.67|0.67|

## Project Structure

```
project-root/
│
├── src/
│ ├── classification_models.py
│ └── data_processing.py
│ └── deep_learning.py
| └── EDA.py
|
│── build.sh
├── classification.ipynb
├── deep_models.ipynb
├── requirements.txt
└── README.md
```
## Installation 
```
chmod +x build.sh && ./build.sh
```
