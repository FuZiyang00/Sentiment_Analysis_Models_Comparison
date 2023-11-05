import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import plotly.express as px
from wordcloud import WordCloud, STOPWORDS
import re 

class EDA:
    def __init__(self, data):
        self.data = data
    
    def dataset_info(self):
        shape = self.data.shape
        duplicates = self.data.duplicated().sum()
        info = self.data.info()
        return shape, duplicates, info
    
    def summary_statistics(self, column):
        statistics = self.data[column].describe().round(2)
        fig = px.histogram(self.data, x=column)
        return statistics, fig 

    def variables_relationship(self, column): 
        self.data["reviews_lenght"] = self.data[column].apply(len)
        line_plot = px.line(self.data, x="reviews_lenght",
                             y=column, title='reviews lenght and rating')
        scatter_plot = px.scatter(self.data, x="reviews_lenght",
                                  y=column, color=column)
        return line_plot, scatter_plot

 
