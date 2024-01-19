import plotly.express as px
import os

class EDA:
    def __init__(self, data):
        self.data = data
        self.plot_folder = os.path.join(os.path.dirname(__file__), '..', 'plot')
        # Create the 'plot' folder if it doesn't exist
        os.makedirs(self.plot_folder, exist_ok=True)
    
    def _save_plot(self, fig, filename):
        filepath = os.path.join(self.plot_folder, f'{filename}.html')
        fig.write_html(filepath)
        print(f'Plot saved at: {filepath}')

    
    def dataset_info(self, eda_file):
        shape = self.data.shape
        duplicates = self.data.duplicated().sum()
        info = self.data.info()

        # writing the dataset information 
        with open(eda_file, 'a') as f: 
            print(f"Dataset shape {shape}:", file=f) # print = write (specify where to print)
            print("\n", file=f)
            print(f"Duplicates: {duplicates}", file=f)
            print("\n", file=f)
            print(f"COlumns info: {info}", file = f)
            print("\n", file=f)
        
    def summary_statistics(self, column, eda_file):
        statistics = self.data[column].describe()
        
        # Write statistics to the output file
        with open(eda_file, 'a') as f:
            print(f"Summary Statistics for {column}:", file=f)
            print(statistics, file=f)
            print("\n", file=f)
        
        # Create and save the histogram figure
        fig = px.histogram(self.data, x=column)
        self._save_plot(fig, f'summary_statistics_{column}')

    def variables_relationship(self, reviews, rating): 
        # investigating over possible relationships between reviews lenght 
        # and the Y (rating)

        self.data["reviews_lenght"] = self.data[reviews].apply(len)
        scatter = 'scatterplot'
        # scatter plot 
        scatter_plot = px.scatter(self.data, x="reviews_lenght",
                                  y=rating)
        
        self._save_plot(scatter_plot, f'{scatter}')
    
    @staticmethod
    def labelling(x):
        if x == 3:
            return "Neutral"
        elif x<3:
            return "Negative"
        else:
            return "Positive"