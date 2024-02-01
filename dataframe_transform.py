import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC
from dataframe_info import DataFrameInfo as info
from plotter import Plotter as plotter


np.random.seed(123)

# Transforming dataframe to edit columns with missing data
class DataFrameTransform:

    # Removing columns which contain null/missing values
    def remove_null_columns(self, DataFrame: pd.DataFrame, column_name):
        DataFrame.drop(column_name, axis=1, inplace=True)
        return DataFrame

    # Removing rows in dataframe where data points are null in a specified column
    def remove_null_rows(self, DataFrame: pd.DataFrame, column_name):
        DataFrame.dropna(subset=column_name, inplace=True)
        return DataFrame
    
    # Replace null values with the median value
    def fill_median(self, DataFrame: pd.DataFrame, column_name)
        DataFrame[column_name].fillna(DataFrame[column_name].median(numeric_only=True), inplace=True)
        return DataFrame
    
    # Replace null values with the mean value
    def fill_mean(self, DataFrame: pd.DataFrame, column_name):
        DataFrame[column_name].fillna(DataFrame[column_name].mean(numeric_only=True, skipna=True), inplace=True)
        return DataFrame
    
    # Attribute null values in a numerical column based on a linear regression model
    def linear_regression_fill(self, DataFrame: pd.DataFrame, column_to_fill: str, training_features: list = None, score: bool = False, check_distribution: bool = False):
        
        # Plotting histogram
        if check_distribution == True:
            print(f'\n({column_to_fill}) Initial Distribution:\n')
            plotter.histogram(self, DataFrame, column_to_fill)

        if training_features == None: # If no training features are provided
            x = DataFrame.drop(info.get_null_columns(self, DataFrame), axis=1) 
        else:  # If training features are provided
            x = DataFrame[training_features] 
        y = DataFrame[column_to_fill]

        # Encodeing string columns to numeric type 
        object_columns = x.select_dtypes(include=['object']).columns.tolist() 
        x[object_columns] = x[object_columns].astype('category') 
        x[object_columns] = x[object_columns].apply(lambda x: x.cat.codes) 

        # Encoding date columns to numeric type
        date_columns = x.select_dtypes(include=['period[M]']).columns.tolist() 
        x[date_columns] = x[date_columns].astype('category') 
        x[date_columns] = x[date_columns].apply(lambda x: x.cat.codes) 


        x_train = x[~y.isna()] 
        y_train = y[~y.isna()] 

        x_test = x[y.isna()] # Testing input data

        # Train Linear Regression Model:
        model = LinearRegression()
        model.fit(x_train, y_train)

        # Run model and impute null values with predicted values
        prediction = model.predict(x_test)
        DataFrame[column_to_fill].loc[y.isna()] = prediction
        
        if check_distribution == True:
            print(f'\n({column_to_fill}) Final Distribution:\n')
            plotter.histogram(self, DataFrame, column_to_fill) # Plots histogram to display distribution

        if score == True:
            print(f'\nScore: {round(model.score(x_train, y_train),2)}') # Provides an accuracy score for the model

        return DataFrame

    # Applying Box-Cox transformation to normalise a column 
    def box_cox_transform(self, DataFrame: pd.DataFrame, column_name: str):
        boxcox_column = stats.boxcox(DataFrame[column_name])
        boxcox_column = pd.Series(boxcox_column[0])
        return boxcox_column

    # Applying Yeo-Johnson transformation to normalise a column
    def yeo_johnson_transform(self, DataFrame: pd.DataFrame, column_name: str):
        yeojohnson_column = stats.yeojohnson(DataFrame[column_name])
        yeojohnson_column = pd.Series(yeojohnson_column[0])
        return yeojohnson_column

    # Removing rows based on the 'z score' of values in a column
    def drop_outlier_rows(self, DataFrame: pd.DataFrame, column_name: str, z_score_threshold: int):
        mean = np.mean(DataFrame[column_name]) # Finding the mean of the column
        std = np.std(DataFrame[column_name]) # Finding the standard deviation of the column
        z_scores = (DataFrame[column_name] - mean) / std # Finding the 'z score' for each value in the column
        abs_z_scores = pd.Series(abs(z_scores)) # Create series with the values of the 'z_score'
        mask = abs_z_scores < z_score_threshold
        DataFrame = DataFrame[mask] # Keep only rows where the 'z score' is below the threshold       
        return DataFrame
